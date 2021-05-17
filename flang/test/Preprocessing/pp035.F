! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: if(777.eq.777)then
! CHECK: print*,'pp035.F no: ',777
* #if 2 .LT. 3 works
      integer, parameter :: KWM = 666
#if 2 .LT. 3
#define KWM 777
#else
#define KWM 667
#endif
      if (KWM .eq. 777) then
        print *, 'pp035.F yes'
      else
        print *, 'pp035.F no: ', KWM
      end if
      end
