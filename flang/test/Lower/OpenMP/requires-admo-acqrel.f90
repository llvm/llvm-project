!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s

module m
!$omp requires atomic_default_mem_order(acq_rel)

contains

subroutine f00(x, v)
  integer :: x, v
!CHECK: omp.atomic.read %{{[ %#=0-9]+}} memory_order(acquire)
  !$omp atomic read
    v = x

!CHECK: omp.atomic.write %{{[ %#=0-9]+}} memory_order(release)
  !$omp atomic write
    x = v
end

end module
