! The predefined identifiers omp_initial_device (-1) and omp_invalid_device
! (-2) from the OpenMP 5.2+ specification must be accepted as valid device
! numbers in the DEVICE clause of target constructs.

! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags -fopenmp-version=52
! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags -fopenmp-version=60

program main
  use omp_lib

  real(8) :: arrayA(256)
  integer :: N, dev
  arrayA = 1.414d0
  N = 256

  ! Literal values allowed by the OpenMP 5.2 / 6.0 specification.
  !$omp target device(-1)
  do i = 1, N
     arrayA(i) = 3.14d0
  enddo
  !$omp end target

  !$omp target device(-2)
  do i = 1, N
     arrayA(i) = 3.14d0
  enddo
  !$omp end target

  !$omp target device(0)
  do i = 1, N
     arrayA(i) = 3.14d0
  enddo
  !$omp end target

  ! Using the predefined identifiers from the omp_lib module.
  !$omp target device(omp_initial_device)
  do i = 1, N
     arrayA(i) = 3.14d0
  enddo
  !$omp end target

  !$omp target device(omp_invalid_device)
  do i = 1, N
     arrayA(i) = 3.14d0
  enddo
  !$omp end target

  ! Also accepted on target data and its data motion variants.
  !$omp target data map(to: arrayA) device(omp_initial_device)
  !$omp end target data

  !$omp target data map(to: arrayA) device(omp_invalid_device)
  !$omp end target data

  !$omp target enter data map(alloc: arrayA) device(omp_initial_device)
  !$omp target enter data map(alloc: arrayA) device(omp_invalid_device)

  !$omp target exit data map(delete: arrayA) device(omp_initial_device)
  !$omp target exit data map(delete: arrayA) device(omp_invalid_device)

  !$omp target update to(arrayA) device(omp_initial_device)
  !$omp target update to(arrayA) device(omp_invalid_device)

  ! Runtime-determined values pass the semantic check.
  dev = -1
  !$omp target device(dev)
  do i = 1, N
     arrayA(i) = 3.14d0
  enddo
  !$omp end target

  ! Values below -2 are still rejected.
  !ERROR: The device expression of the DEVICE clause must be a non-negative integer expression, 'omp_initial_device' (-1), or 'omp_invalid_device' (-2)
  !$omp target device(-3)
  do i = 1, N
     arrayA(i) = 3.14d0
  enddo
  !$omp end target

end program main
