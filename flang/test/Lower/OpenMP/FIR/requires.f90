! RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s
! RUN: bbc -fopenmp -emit-fir %s -o - | FileCheck %s
! RUN: bbc -fopenmp -fopenmp-is-target-device -emit-fir %s -o - | FileCheck %s

! This test checks the lowering of requires into MLIR

!CHECK:      module attributes {
!CHECK-SAME: omp.requires = #omp<clause_requires reverse_offload|unified_shared_memory>
program requires
  !$omp requires unified_shared_memory reverse_offload atomic_default_mem_order(seq_cst)
  !$omp target
  !$omp end target
end program requires
