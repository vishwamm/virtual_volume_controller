import cv2
import mediapipe as mp
import time


class handdetector():
    def __init__(self,mode=False,maxhands=2):
        self.mode=mode
        self.maxhands=maxhands
        self.mphands=mp.solutions.hands
        self.hands=self.mphands.Hands(self.mode,self.maxhands)
        self.mpDraw=mp.solutions.drawing_utils

    def  findHands(self,image,draw=True):
        self.imgRGB=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results=self.hands.process(self.imgRGB)
        if self.results.multi_hand_landmarks:
            for  handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image,handLms,self.mphands.HAND_CONNECTIONS)
        return image
    def findposition(self,image,handno=0,draw=True):
        lmlist=[]
        if self.results.multi_hand_landmarks:
            myhand=self.results.multi_hand_landmarks[handno]
            for  id, lm in enumerate(myhand.landmark):
                #print(lm)
                h,w,c=self.imgRGB.shape
                x,y=int(lm.x*w),int(lm.y*h)
                lmlist.append([id,x,y])
                if draw:
                    cv2.circle(image,(int(x),int(y)),8,(255,0,255),cv2.FILLED)
        return lmlist

                
    
def main():
    cap=cv2.VideoCapture(0)
    cTime=0
    pTime=0
    detector=handdetector()
    while True:
        success,image=cap.read()
        image=detector.findHands(image)
        lmlist=detector.findposition(image)
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        #image,fps,size,font,scale,color,thickness
        cv2.putText(image,"FPS:"+str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),4)
        cv2.imshow("Image", image)
        cv2.waitKey(1)

if __name__=="__main__":
    main()

