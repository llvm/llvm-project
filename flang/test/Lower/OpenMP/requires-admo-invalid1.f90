!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s

module m
!$omp requires atomic_default_mem_order(acquire)

contains

subroutine f00(x, v)
  integer :: x, v
!CHECK: omp.atomic.write %{{[ %#=0-9]+}} memory_order(relaxed)
  !$omp atomic write
    x = v
end

end module

