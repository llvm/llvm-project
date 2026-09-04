!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

subroutine f00(a)
  integer :: a(*)
  ! No diagnostic expected, assumed-size arrays are allowed on USE_DEVICE_ADDR
  ! in 6.0.
  !$omp target_data use_device_addr(a)
  !$omp end target_data
end
