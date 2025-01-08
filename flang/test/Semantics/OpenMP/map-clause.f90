! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52
! Check OpenMP MAP clause validity. Section 5.8.3 OpenMP 5.2.

subroutine sb(arr)
  implicit none
  real(8) :: arr(*)
  real :: a
  integer:: b, c, i
  common /var/ b, c  
  
  !ERROR: Assumed-size whole arrays may not appear on the MAP clause
  !$omp target map(arr)
  do i = 1, 100
     a = 3.14
  enddo
  !$omp end target

  !ERROR: Assumed-size array 'arr' must have explicit final subscript upper bound value
  !$omp target map(arr(:))
  do i = 1, 100
     a = 3.14
  enddo
  !$omp end target

  !$omp target map(arr(3:5))
  do i = 1, 100
     a = 3.14
  enddo
  !$omp end target

 !$omp target map(tofrom: /var/)
   b = 1
   c = 2
 !$omp end target
end subroutine

subroutine sb1
  integer :: xx
  integer :: a
  !ERROR: Name 'xx' should be a mapper name
  !$omp target map(mapper(xx), from:a)
  !$omp end target
end subroutine sb1
