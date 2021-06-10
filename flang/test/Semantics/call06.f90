! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
! Test 15.5.2.6 constraints and restrictions for ALLOCATABLE
! dummy arguments.

module m

  real, allocatable :: cov[:], com[:,:]

 contains

  subroutine s01(x)
    real, allocatable :: x
  end subroutine
  subroutine s02(x)
    real, allocatable :: x[:]
  end subroutine
  subroutine s03(x)
    real, allocatable :: x[:,:]
  end subroutine
  subroutine s04(x)
    real, allocatable, intent(in) :: x
  end subroutine
  subroutine s05(x)
    real, allocatable, intent(out) :: x
  end subroutine
  subroutine s06(x)
    real, allocatable, intent(in out) :: x
  end subroutine
  function allofunc()
    real, allocatable :: allofunc
  end function

  subroutine test(x)
    real :: scalar
    real, allocatable, intent(in) :: x
    !ERROR: ALLOCATABLE dummy argument 'x=' must be associated with an ALLOCATABLE actual argument
    call s01(scalar)
    !ERROR: ALLOCATABLE dummy argument 'x=' must be associated with an ALLOCATABLE actual argument
    call s01(1.)
    !ERROR: ALLOCATABLE dummy argument 'x=' must be associated with an ALLOCATABLE actual argument
    call s01(allofunc()) ! subtle: ALLOCATABLE function result isn't
    call s02(cov) ! ok
    call s03(com) ! ok
    !ERROR: ALLOCATABLE dummy argument 'x=' has corank 1 but actual argument has corank 2
    call s02(com)
    !ERROR: ALLOCATABLE dummy argument 'x=' has corank 2 but actual argument has corank 1
    call s03(cov)
    call s04(cov[1]) ! ok
    !ERROR: ALLOCATABLE dummy argument 'x=' must have INTENT(IN) to be associated with a coindexed actual argument
    call s01(cov[1])
    !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'x=' must be definable
    call s05(x)
    !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'x=' must be definable
    call s06(x)
  end subroutine
end module
