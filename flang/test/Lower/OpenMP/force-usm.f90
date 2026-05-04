! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-force-usm %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-is-target-device -fopenmp-force-usm %s -o - | FileCheck %s
! RUN: bbc -fopenmp -fopenmp-force-usm -emit-hlfir %s -o - | FileCheck %s
! RUN: bbc -fopenmp -fopenmp-is-target-device -fopenmp-force-usm -emit-hlfir %s -o - | FileCheck %s

! This test checks the addition of requires unified_shared_memory when
! -fopenmp-force-usm is set

!CHECK:      module attributes {
!CHECK-SAME: omp.requires = #omp<clause_requires unified_shared_memory>
program requires
end program requires
