! This test checks the lowering of REQUIRES inside of an unnamed BLOCK DATA.
! The symbol of the `symTab` scope of the `BlockDataUnit` PFT node is null in
! this case, resulting in the inability to store the REQUIRES flags gathered in
! it.

! RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s
! RUN: bbc -fopenmp -emit-fir %s -o - | FileCheck %s
! RUN: bbc -fopenmp -fopenmp-is-target-device -emit-fir %s -o - | FileCheck %s
! XFAIL: *

!CHECK:         module attributes {
!CHECK-SAME:    omp.requires = #omp<clause_requires unified_shared_memory>
block data
  !$omp requires unified_shared_memory
  integer :: x
  common /block/ x
  data x / 10 /
end

subroutine f
  !$omp declare target
end subroutine f
