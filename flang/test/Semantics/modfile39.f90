! RUN: %S/test_modfile.sh %s %t %flang_fc1
! REQUIRES: shell
! Resolution of specification expression references to generic interfaces
! that resolve to private specific functions.

module m1
  interface gen
    module procedure priv
  end interface
  private :: priv
 contains
  pure integer function priv(n)
    integer, intent(in) :: n
    priv = n
  end function
end module
!Expect: m1.mod
!module m1
!interface gen
!procedure::priv
!end interface
!private::priv
!contains
!pure function priv(n)
!integer(4),intent(in)::n
!integer(4)::priv
!end
!end

module m2
  use m1
 contains
  subroutine s(a)
    real :: a(gen(1))
  end subroutine
end module
!Expect: m2.mod
!module m2
!use m1,only:gen
!use m1,only:m1$priv=>priv
!private::m1$priv
!contains
!subroutine s(a)
!real(4)::a(1_8:int(m1$priv(1_4),kind=8))
!end
!end

use m2
end
