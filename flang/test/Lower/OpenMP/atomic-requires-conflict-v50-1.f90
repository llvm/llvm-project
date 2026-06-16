!RUN: bbc %openmp_flags -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s
!RUN: %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=50 %s -o - | FileCheck %s

!CHECK: omp.atomic.read %{{[0-9]+}}#0 = %{{[0-9]+}}#0 memory_order(acquire)

subroutine f00(x, v)
  integer :: x, v
  !$omp requires atomic_default_mem_order(acq_rel)
  !$omp atomic read
    v = x
end
