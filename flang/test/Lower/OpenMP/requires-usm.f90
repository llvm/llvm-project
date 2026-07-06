! RUN: split-file %s %t

! Verify that we pick up requires USM and apply it correctly when it is specified
! outside of the program.

! RUN: %flang_fc1 -emit-hlfir -fopenmp %t/requires-usm.f90 -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-is-target-device %t/requires-usm.f90 -o - | FileCheck %s
! RUN: bbc -fopenmp -emit-hlfir %t/requires-usm.f90 -o - | FileCheck %s
! RUN: bbc -fopenmp -fopenmp-is-target-device -emit-hlfir %t/requires-usm.f90 -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp %t/requires-usm-subroutine-after.f90 -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-is-target-device %t/requires-usm-subroutine-after.f90 -o - | FileCheck %s
! RUN: bbc -fopenmp -emit-hlfir %t/requires-usm-subroutine-after.f90 -o - | FileCheck %s
! RUN: bbc -fopenmp -fopenmp-is-target-device -emit-hlfir %t/requires-usm-subroutine-after.f90 -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp %t/requires-usm-program-after.f90 -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-is-target-device %t/requires-usm-program-after.f90 -o - | FileCheck %s
! RUN: bbc -fopenmp -emit-hlfir %t/requires-usm-program-after.f90 -o - | FileCheck %s
! RUN: bbc -fopenmp -fopenmp-is-target-device -emit-hlfir %t/requires-usm-program-after.f90 -o - | FileCheck %s

! CHECK:      module attributes {
! CHECK-SAME: omp.requires = #omp<clause_requires unified_shared_memory>

!--- requires-usm.f90
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

!--- requires-usm-subroutine-after.f90
program main
  implicit none
end program main

subroutine test
  !$omp requires unified_shared_memory
!$omp target
!$omp end target
end subroutine

!--- requires-usm-program-after.f90
subroutine test
end subroutine

program main
  implicit none
!$omp requires unified_shared_memory
!$omp target
!$omp end target
end program main
