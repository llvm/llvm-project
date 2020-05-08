! RUN: bbc -emit-fir -o - %s | FileCheck %s

   character*10 :: exx
   character*20 :: c
   character*30 :: m
   integer*2 :: s
   exx = 'AA'
   c = 'BBBB'
   m = 'CCCCCC'
   s = -13
   ! CHECK: call {{.*}}BeginExternalFormattedInput
   ! CHECK: call {{.*}}EnableHandlers
   ! CHECK: call {{.*}}SetAdvance
   ! CHECK: call {{.*}}InputAscii
   ! CHECK: call {{.*}}GetIoMsg
   ! CHECK: call {{.*}}EndIoStatement
   ! CHECK: fir.select %{{.*}} : index [-2, ^bb4, -1, ^bb3, 0, ^bb1, unit, ^bb2]
   read(*, '(A)', ADVANCE='NO', ERR=10, END=20, EOR=30, IOSTAT=s, IOMSG=m) c
   ! CHECK-LABEL: ^bb1:
   exx = 'Zip'; goto 90
10 exx = 'Err'; goto 90
20 exx = 'End'; goto 90
30 exx = 'Eor'; goto 90
90 print*, exx, c, m, s
end
