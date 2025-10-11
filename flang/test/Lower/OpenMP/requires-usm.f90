! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s
! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s
! RUN: bbc -fopenmp -fopenmp-is-target-device -emit-hlfir %s -o - | FileCheck %s

! Verify that we pick up USM and apply it correctly when it is specified
! outside of the program.

!CHECK:      module attributes {
!CHECK-SAME: omp.requires = #omp<clause_requires unified_shared_memory>
module declare_mod
    implicit none
!$omp requires unified_shared_memory
 contains
end module

program main
 use declare_mod
 implicit none
!$omp target
!$omp end target
end program
