! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: if(777.eq.777)then
! CHECK: print*,'pp032.F no: ',777
* /* C comment */ \ newline erased from #define
      integer, parameter :: KWM = 666
#define KWM 77/* C comment */\
7
      if (KWM .eq. 777) then
        print *, 'pp032.F yes'
      else
        print *, 'pp032.F no: ', KWM
      end if
      end
