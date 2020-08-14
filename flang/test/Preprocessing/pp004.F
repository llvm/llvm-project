! RUN: %f18 -E %s 2>&1 | FileCheck %s
! CHECK: if(kwm.eq.777)then
* KWMs case-sensitive
      integer, parameter :: KWM = 777
#define KWM 666
      if (kwm .eq. 777) then
        print *, 'pp004.F yes'
      else
        print *, 'pp004.F no: ', kwm
      end if
      end
