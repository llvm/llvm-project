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

module mc
  character(len=4), parameter :: cp(2) = ['abcd', 'efgh']
contains
  subroutine takes_c2x2(c)
    character(len=2), intent(in) :: c(2)
  end subroutine
  subroutine takes_c2x3(c)
    character(len=2), intent(in) :: c(3)
  end subroutine
end module

subroutine char_accepted()
  use mc
  ! Character storage sequence association (F'2023 15.5.2.12 p4): the 4
  ! characters remaining from cp(2) exactly fill the 2x2-character dummy.
  call takes_c2x2(cp(2))
end subroutine

subroutine char_short_sequence()
  use mc
  !ERROR: Actual argument has fewer characters remaining in storage sequence (4) than dummy argument 'c=' (6)
  call takes_c2x3(cp(2))
end subroutine

module mdc
  type :: dt
    integer :: i
    character :: c
  end type
  type(dt), parameter :: dp(3) = [dt(1, 'a'), dt(2, 'b'), dt(3, 'c')]
  integer, parameter :: nlb(-1:4) = [1, 2, 3, 4, 5, 6]
contains
  subroutine dt3(x)
    type(dt), intent(in) :: x(3)
  end subroutine
  subroutine dt2(x)
    type(dt), intent(in) :: x(2)
  end subroutine
  subroutine int5(x)
    integer, intent(in) :: x(5)
  end subroutine
  subroutine int6(x)
    integer, intent(in) :: x(6)
  end subroutine
end module

subroutine derived_and_lower_bounds()
  use mdc
  ! Exact-fit boundary cases are accepted.
  call dt3(dp(1))
  call dt2(dp(2))
  ! An element of a named constant with a nondefault lower bound: five
  ! elements remain from nlb(0).
  call int5(nlb(0))
  !ERROR: Actual argument has fewer elements remaining in storage sequence (2) than dummy argument 'x=' array (3)
  call dt3(dp(2))
  !ERROR: Actual argument has fewer elements remaining in storage sequence (5) than dummy argument 'x=' array (6)
  call int6(nlb(0))
end subroutine

subroutine generic_vs_intrinsic()
  ! A user generic named like an intrinsic still resolves correctly with
  ! retained named-constant arguments, and intrinsic uses see the values
  ! of named-constant elements (DIM=, KIND=).
  interface sum
    procedure mysum
  end interface
  integer, parameter :: gp2(-1:2) = [1, 2, 3, 4]
  integer, parameter :: kk(2) = [4, 8]
  integer :: a(2, 2)
  integer :: r(2)
  a = reshape(gp2, [2, 2])
  print *, sum(gp2, 7)          ! user generic: extra scalar argument
  r = sum(array=a, dim=gp2(0))  ! intrinsic: DIM= from a named-constant element
  print *, r, kind(int(1, kind=kk(2)))
contains
  integer function mysum(x, y)
    integer, intent(in) :: x(4), y
    mysum = x(1) + x(2) + x(3) + x(4) + y
  end function
end subroutine

subroutine value_dummy_not_yet()
  ! An array VALUE dummy needs a temporary covering the whole storage
  ! sequence, which lowering does not create yet (llvm-project#224636):
  ! the named-constant element form stays rejected for now, while whole
  ! named-constant arrays and variable elements are unaffected.
  use m
  integer :: v(4)
  interface
    subroutine byval3(x)
      integer, value :: x(3)
    end subroutine
    subroutine byval4(x)
      integer, value :: x(4)
    end subroutine
  end interface
  !ERROR: not yet implemented: sequence association of a named constant array element with a VALUE dummy argument 'x=' array
  call byval3(gp(2))
  call byval4(gp)   ! whole array: accepted
  call byval3(v(2)) ! variable element: accepted (preexisting behavior)
end subroutine
