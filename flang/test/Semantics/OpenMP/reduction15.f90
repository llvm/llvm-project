!RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=50

!Ref: [5.0:297:28-29]
! If a list item is an array section or an array element, its base expression
! must be a base language identifier.

module m
  type t
    integer :: a(10)
  end type

contains

  subroutine f00
    type(t) :: x
  !ERROR: The base expression of an array element or section in REDUCTION clause must be an identifier
  !$omp do reduction (+ : x%a(2))
    do i = 1, 10
    end do
  !$omp end do
  end subroutine

  subroutine f01
    type(t) :: x
  !ERROR: The base expression of an array element or section in REDUCTION clause must be an identifier
  !$omp do reduction (+ : x%a(1:10))
    do i = 1, 10
    end do
  !$omp end do
  end subroutine
end

