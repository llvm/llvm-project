! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: if (777 .eq. 777) then
* KLM rescan
      integer, parameter :: KWM = 666, KWM2 = 667
#define KWM2 777
#define KWM KWM2
      if (KWM .eq. 777) then
        print *, 'pp017.F yes'
      else
        print *, 'pp017.F no: ', KWM
      end if
      end
