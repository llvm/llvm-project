! RUN: %python %S/test_errors.py %s %flang_fc1
! Named constants (PARAMETER) as actual arguments: element sequence
! association is accepted (F'2023 15.5.2.12), short sequences and attempts
! to modify the dummy are still diagnosed, and intrinsic argument checks
! that need constant values still fire.
module m
  integer, parameter :: gp(4) = [1, 2, 3, 4]
  integer, parameter :: d1(1) = [5]
contains
  subroutine expl3(x)
    integer, intent(in) :: x(3)
  end subroutine
  subroutine expl4(x)
    integer, intent(in) :: x(4)
  end subroutine
  subroutine modifies(x)
    integer, intent(inout) :: x(3)
  end subroutine
  subroutine outputs(x)
    integer, intent(out) :: x(3)
  end subroutine
end module

subroutine accepted()
  use m
  ! Element of a named constant to an explicit-shape dummy whose extent
  ! matches the remaining sequence: conforming, no error.
  call expl3(gp(2))
end subroutine

subroutine short_sequence()
  use m
  !ERROR: Actual argument has fewer elements remaining in storage sequence (3) than dummy argument 'x=' array (4)
  call expl4(gp(2))
end subroutine

subroutine not_definable()
  use m
  !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'x=' is not definable
  !BECAUSE: 'gp' is not a variable
  call modifies(gp(2))
  !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'x=' is not definable
  !BECAUSE: 'gp' is not a variable
  call outputs(gp(2))
end subroutine

subroutine intrinsic_dim_check()
  use m
  real :: a(2, 3)
  ! A named-constant DIM= argument out of range must still be diagnosed
  ! (the designator is folded back to a value for intrinsic checking).
  !ERROR: The value of DIM= (5) may not be greater than 2
  print *, sum(a, dim=d1(1))
end subroutine
