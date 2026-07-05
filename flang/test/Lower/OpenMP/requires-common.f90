! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s
! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s
! RUN: bbc -fopenmp -fopenmp-is-target-device -emit-hlfir %s -o - | FileCheck %s

! This test checks the lowering of requires into MLIR

!CHECK:      module attributes {
!CHECK-SAME: omp.requires = #omp<clause_requires unified_shared_memory>
block data init
  !$omp requires unified_shared_memory
  integer :: x
  common /block/ x
  data x / 10 /
end

subroutine f
  !$omp declare target
end subroutine f
