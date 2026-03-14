! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK-NOT:z = 111
! CHECK:warning: Statement should not begin with a continuation line
! CHECK:j=111+444
* #define directive amid continuations.
      integer, parameter :: KWM = 222
      integer, parameter :: z = KWM
#define KWM 111
     ,j=KWM+444
      if (z .EQ. 222 .AND. j .EQ. 555) then
        print *, 'yes'
      else
        print *, 'no', z, _4
      end if
      end
