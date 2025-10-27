!RUN: bbc %openmp_flags -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s
!RUN: %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=50 %s -o - | FileCheck %s

!CHECK: omp.atomic.update memory_order(relaxed)

subroutine f05(x, v)
  integer :: x, v
  !$omp requires atomic_default_mem_order(acq_rel)
  !$omp atomic update
    x = x + 1
end
