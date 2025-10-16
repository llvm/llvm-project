! RUN: not %flang_fc1 -fopenmp -fopenmp-version=50 %s 2>&1 | FileCheck %s
program main
  integer, parameter :: n = 256
  real(8) :: a(256)
  !$omp target map(mapper(xx), from:a)
  do i=1,n
     a(i) = 4.2
  end do
  !$omp end target
end program main

! CHECK: error: '{{.*}}' not declared
