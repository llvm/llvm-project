! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=50
! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51

program main
  real(8) :: arrayA(256)
  integer :: N
  arrayA = 1.414d0
  N = 256

  !ERROR: The device expression of the DEVICE clause must be a positive integer expression
  !$omp target device(-1)
  do i = 1, N
     arrayA(i) = 3.14d0
  enddo
  !$omp end target

  !ERROR: The device expression of the DEVICE clause must be a positive integer expression
  !$omp target device(-2)
  do i = 1, N
     arrayA(i) = 3.14d0
  enddo
  !$omp end target

  !ERROR: The device expression of the DEVICE clause must be a positive integer expression
  !$omp target device(-3)
  do i = 1, N
     arrayA(i) = 3.14d0
  enddo
  !$omp end target

  !$omp target device(0)
  do i = 1, N
     arrayA(i) = 3.14d0
  enddo
  !$omp end target
end program main
