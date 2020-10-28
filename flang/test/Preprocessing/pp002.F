! RUN: %f18 -E %s 2>&1 | FileCheck %s
! CHECK: if(kwm.eq.777)then
* #undef
      integer, parameter :: KWM = 777
#define KWM 666
#undef KWM
      if (KWM .eq. 777) then
        print *, 'pp002.F yes'
      else
        print *, 'pp002.F no: ', KWM
      end if
      end
