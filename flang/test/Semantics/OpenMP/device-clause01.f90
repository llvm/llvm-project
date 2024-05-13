! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! OpenMP Version 5.2
! 13.2 Device clause

subroutine foo

  integer :: a

  !$omp target device(ancestor:0)
  !$omp end target
  !$omp target device(device_num:0)
  !$omp end target
  
  !ERROR: The ANCESTOR device-modifier must not appear on the DEVICE clause on any directive other than the TARGET construct. Found on TARGET DATA construct.
  !$omp target data device(ancestor:0) map(tofrom:a)
  !$omp end target data
  !$omp target data device(device_num:0) map(tofrom:a)
  !$omp end target data

  
  !ERROR: The ANCESTOR device-modifier must not appear on the DEVICE clause on any directive other than the TARGET construct. Found on TARGET ENTER DATA construct.
  !$omp target enter data device(ancestor:0) map(to:a)
  !$omp target exit data map(from:a)
  !$omp target enter data device(device_num:0) map(to:a)
  !$omp target exit data map(from:a)

  !ERROR: The ANCESTOR device-modifier must not appear on the DEVICE clause on any directive other than the TARGET construct. Found on TARGET UPDATE construct.
  !$omp target update device(ancestor:0) to(a)
  !$omp target update device(device_num:0) to(a)

end subroutine foo
