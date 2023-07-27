! RUN: %python %S/test_errors.py %s %flang_fc1
! Ensure that DO CONCURRENT purity checks apply to specific procedures
! in the case of calls to generic interfaces.
module m
  interface purity
    module procedure :: ps, ips
  end interface
  type t
   contains
    procedure :: pb, ipb
    generic :: purity => pb, ipb
  end type
 contains
  pure subroutine ps(n)
    integer, intent(in) :: n
  end subroutine
  impure subroutine ips(a)
    real, intent(in) :: a
  end subroutine
  pure subroutine pb(x,n)
    class(t), intent(in) :: x
    integer, intent(in) :: n
  end subroutine
  impure subroutine ipb(x,n)
    class(t), intent(in) :: x
    real, intent(in) :: n
  end subroutine
end module

program test
  use m
  type(t) :: x
  do concurrent (j=1:1)
    call ps(1) ! ok
    call purity(1) ! ok
    !ERROR: Impure procedure 'ips' may not be referenced in DO CONCURRENT
    call purity(1.)
    !ERROR: Impure procedure 'ips' may not be referenced in DO CONCURRENT
    call ips(1.)
    call x%pb(1) ! ok
    call x%purity(1) ! ok
    !ERROR: Impure procedure 'ipb' may not be referenced in DO CONCURRENT
    call x%purity(1.)
    !ERROR: Impure procedure 'ipb' may not be referenced in DO CONCURRENT
    call x%ipb(1.)
  end do
end program
