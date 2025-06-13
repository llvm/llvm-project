! This test checks lowering of the parallel master combined construct.

! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPparallel_master
subroutine parallel_master(x)
  integer :: x
  !CHECK: omp.parallel {
  !CHECK: omp.master {
  !$omp parallel master
  x = 1
  !$omp end parallel master
  !CHECK: }
  !CHECK: }
end subroutine parallel_master
