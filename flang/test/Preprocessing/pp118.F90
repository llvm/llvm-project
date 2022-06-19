! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: if (KWM2 .eq. 777) then
! KWM rescan with #undef, proving rescan after expansion
      integer, parameter :: KWM2 = 777, KWM = 667
#define KWM2 666
#define KWM KWM2
#undef KWM2
      if (KWM .eq. 777) then
        print *, 'pp118.F90 yes'
      else
        print *, 'pp118.F90 no: ', KWM
      end if
      end
