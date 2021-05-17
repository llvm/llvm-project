! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: if(777.eq.777)then
* keyword macros
      integer, parameter :: KWM = 666
#define KWM 777
      if (KWM .eq. 777) then
        print *, 'pp001.F yes'
      else
        print *, 'pp001.F no: ', KWM
      end if
      end
