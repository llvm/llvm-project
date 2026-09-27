!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52

subroutine f00
  implicit none
  integer, allocatable :: a(:)

  !$omp allocators allocate(a)
!ERROR: The body of an ALLOCATORS construct should be an ALLOCATE statement
  continue
end

subroutine f01
  implicit none
  integer, allocatable :: a(:)

!ERROR: The body of an ALLOCATORS construct should be an ALLOCATE statement
  !$omp allocators allocate(a)
  !$omp end allocators
end

subroutine f02
  implicit none
  integer, allocatable :: a(:)

!ERROR: The body of an ALLOCATORS construct should be an ALLOCATE statement
  !$omp allocators allocate(a)
end
