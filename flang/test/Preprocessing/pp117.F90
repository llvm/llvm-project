! RUN: %f18 -E %s 2>&1 | FileCheck %s
! CHECK: if(777 .eq. 777) then
! KWM rescan
      integer, parameter :: KWM = 666, KWM2 = 667
#define KWM2 777
#define KWM KWM2
      if (KWM .eq. 777) then
        print *, 'pp117.F90 yes'
      else
        print *, 'pp117.F90 no: ', KWM
      end if
      end
