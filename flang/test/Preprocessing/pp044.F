! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK-NOT:z = 111
* #define directive amid continuations
      integer, parameter :: KWM = 222, KWM111 = 333, KWM222 = 555
      integer, parameter :: KWMKWM = 333
      integer, parameter :: z = KWM
#define KWM 111
     +KWM+444
      if (z .EQ. 777) then
        print *, 'yes'
      else
        print *, 'no', z
      end if
      end
