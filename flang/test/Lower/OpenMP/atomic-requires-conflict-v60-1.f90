!RUN: bbc %openmp_flags -fopenmp-version=60 -emit-hlfir %s -o - | FileCheck %s
!RUN: %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=60 %s -o - | FileCheck %s

!CHECK: omp.atomic.read %{{[0-9]+}}#0 = %{{[0-9]+}}#0 memory_order(relaxed)

subroutine f01(x, v)
  integer :: x, v
  !$omp requires atomic_default_mem_order(release)
  !$omp atomic read
    v = x
end
