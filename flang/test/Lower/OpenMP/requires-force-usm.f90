! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-force-usm %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-is-target-device -fopenmp-force-usm %s -o - | FileCheck %s
! RUN: bbc -fopenmp -fopenmp-force-usm -emit-hlfir %s -o - | FileCheck %s
! RUN: bbc -fopenmp -fopenmp-is-target-device -fopenmp-force-usm -emit-hlfir %s -o - | FileCheck %s

! This test checks the addition of requires unified_shared_memory when
! -fopenmp-force-usm is set, even when other requires directives are present

!CHECK:      module attributes {
!CHECK-SAME: omp.requires = #omp<clause_requires reverse_offload|unified_shared_memory>
program requires
  !$omp requires reverse_offload
  !$omp target
  !$omp end target
end program requires
