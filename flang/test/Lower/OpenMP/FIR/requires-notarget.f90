! RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s
! RUN: bbc -fopenmp -emit-fir %s -o - | FileCheck %s
! RUN: bbc -fopenmp -fopenmp-is-target-device -emit-fir %s -o - | FileCheck %s

! This test checks that requires lowering into MLIR skips creating the
! omp.requires attribute with target-related clauses if there are no device
! functions in the compilation unit

!CHECK:      module attributes {
!CHECK-NOT:  omp.requires
program requires
  !$omp requires unified_shared_memory reverse_offload atomic_default_mem_order(seq_cst)
end program requires
