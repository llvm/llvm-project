! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! Check OpenMP MAP clause validity. Section 5.8.3 OpenMP 5.2.

subroutine sb(arr)
  real(8) :: arr(*)
  real :: a

  !ERROR: Assumed-size whole arrays may not appear on the MAP clause
  !$omp target map(arr)
  do i = 1, 100
     a = 3.14
  enddo
  !$omp end target

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
end subroutine
