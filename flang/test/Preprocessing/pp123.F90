! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: ch = 3HKWM
! KWM NOT expanded in Hollerith literal
#define KWM 666
#define HKWM 667
      character(len=3) :: ch
      ch = 3HKWM
      if (ch .eq. 'KWM') then
        print *, 'pp123.F90 yes'
      else
        print *, 'pp123.F90 no: ', ch
      end if
      end
