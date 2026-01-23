!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

module m00
implicit none
integer :: x = 1
!ERROR: GROUPPRIVATE argument cannot be declared with an initializer
!$omp groupprivate(x)
!ERROR: GROUPPRIVATE argument should be a variable or a named common block
!$omp groupprivate(f00)

contains
subroutine f00
  implicit none
  integer, save :: y
  associate (z => y)
  block
    !ERROR: GROUPPRIVATE argument cannot be an ASSOCIATE name
    !$omp groupprivate(z)
  end block
  end associate
end
end module

module m01
implicit none
integer :: x, y
common /some_block/ x
!ERROR: GROUPPRIVATE argument cannot be a member of a common block
!$omp groupprivate(x)

contains
subroutine f01
  implicit none
  integer :: z
  !ERROR: GROUPPRIVATE argument variable must be declared in the same scope as the construct on which it appears
  !$omp groupprivate(y)
  !ERROR: GROUPPRIVATE argument variable must be declared in the module scope or have SAVE attribute
  !$omp groupprivate(z)
end
end module

module m02
implicit none
integer :: x(10)[*]
!ERROR: GROUPPRIVATE argument cannot be a coarray
!$omp groupprivate(x)
end module
