! RUN: %f18 -E %s 2>&1 | FileCheck %s
! CHECK: if(77 7.eq.777)then
* \ newline allowed in #define
      integer, parameter :: KWM = 666
#define KWM 77\
 7
      if (KWM .eq. 777) then
        print *, 'pp029.F yes'
      else
        print *, 'pp029.F no: ', KWM
      end if
      end
