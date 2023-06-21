! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
! Test the RelaxedIntentInChecking extension
module m
 contains
  subroutine intentInUnlimited(x)
    class(*), dimension(..), pointer, intent(in) :: x
  end
  subroutine intentInOutUnlimited(x)
    class(*), dimension(..), pointer, intent(in out) :: x
  end
  subroutine test
    integer, target :: scalar
    real, pointer :: arrayptr(:)
    class(*), pointer :: unlimited(:)
    call intentInUnlimited(scalar)
    !ERROR: Actual argument associated with POINTER dummy argument 'x=' must also be POINTER unless INTENT(IN)
    call intentInOutUnlimited(scalar)
    !PORTABILITY: If a POINTER or ALLOCATABLE dummy or actual argument is unlimited polymorphic, both should be so
    call intentInUnlimited(arrayptr)
    !ERROR: If a POINTER or ALLOCATABLE dummy or actual argument is unlimited polymorphic, both must be so
    call intentInOutUnlimited(arrayptr)
    call intentInUnlimited(unlimited) ! ok
    call intentInOutUnlimited(unlimited) ! ok
  end
end
