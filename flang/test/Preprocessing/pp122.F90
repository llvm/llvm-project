! RUN: %f18 -E %s 2>&1 | FileCheck %s
! CHECK: ch = "KWM"
! KWM NOT expanded in "literal"
#define KWM 666
      character(len=3) :: ch
      ch = "KWM"
      if (ch .eq. 'KWM') then
        print *, 'pp122.F90 yes'
      else
        print *, 'pp122.F90 no: ', ch
      end if
      end
