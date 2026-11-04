!RUN: bbc %openmp_flags -fopenmp-version=60 -emit-hlfir %s -o - | FileCheck %s
!RUN: %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=60 %s -o - | FileCheck %s

!CHECK: omp.atomic.write %{{[0-9]+}}#0 = %{{[0-9]+}} memory_order(relaxed)

subroutine f02(x, v)
  integer :: x, v
  !$omp requires atomic_default_mem_order(acquire)
  !$omp atomic write
    x = v
end
