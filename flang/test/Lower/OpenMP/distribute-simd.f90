! This test checks lowering of OpenMP DISTRIBUTE SIMD composite constructs.

! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPdistribute_simd_aligned(
subroutine distribute_simd_aligned(A)
  use iso_c_binding
  type(c_ptr) :: A
  
  !$omp teams

  ! CHECK:      omp.distribute
  ! CHECK-NOT:  aligned({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: aligned({{.*}})
  !$omp distribute simd aligned(A:256)
  do index_ = 1, 10
    call c_test_call(A)
  end do
  !$omp end distribute simd

  !$omp end teams
end subroutine distribute_simd_aligned

! CHECK-LABEL: func.func @_QPdistribute_simd_safelen(
subroutine distribute_simd_safelen()
  !$omp teams

  ! CHECK:      omp.distribute
  ! CHECK-NOT:  safelen({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: safelen({{.*}})
  !$omp distribute simd safelen(4)
  do index_ = 1, 10
  end do
  !$omp end distribute simd

  !$omp end teams
end subroutine distribute_simd_safelen

! CHECK-LABEL: func.func @_QPdistribute_simd_simdlen(
subroutine distribute_simd_simdlen()
  !$omp teams

  ! CHECK:      omp.distribute
  ! CHECK-NOT:  simdlen({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: simdlen({{.*}})
  !$omp distribute simd simdlen(4)
  do index_ = 1, 10
  end do
  !$omp end distribute simd

  !$omp end teams
end subroutine distribute_simd_simdlen
