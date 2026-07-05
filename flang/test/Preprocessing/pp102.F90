! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: if (KWM .eq. 777) then
! #undef
      integer, parameter :: KWM = 777
#define KWM 666
#undef KWM
      if (KWM .eq. 777) then
        print *, 'pp102.F90 yes'
      else
        print *, 'pp102.F90 no: ', KWM
      end if
      end
