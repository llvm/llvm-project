! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags -fopenmp-version=52
! OpenMP Version 5.2

program my_fib
  integer :: n = 8
  !$omp declare target(fib)
  !$omp target
    call fib(n)
  !$omp end target
end program my_fib

subroutine fib(n)
  integer :: n
  !$omp declare target
  print *, "hello from fib"
end subroutine fib
