!RUN: bbc %openmp_flags -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s
!RUN: %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=50 %s -o - | FileCheck %s

!CHECK: omp.atomic.capture memory_order(relaxed)

subroutine f06(x, v)
  integer :: x, v
  !$omp requires atomic_default_mem_order(acquire)
  !$omp atomic update capture
    v = x
    x = x + 1
  !$omp end atomic
end
