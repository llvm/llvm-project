! This test checks lowering of OpenMP DO SIMD composite constructs.

! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPdo_simd_aligned(
subroutine do_simd_aligned(A)
  use iso_c_binding
  type(c_ptr) :: A
  
  ! CHECK:      omp.wsloop
  ! CHECK-NOT:  aligned({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: aligned({{.*}})
  !$omp do simd aligned(A:256)
    do index_ = 1, 10
      call c_test_call(A)
    end do
  !$omp end do simd
end subroutine do_simd_aligned

! CHECK-LABEL: func.func @_QPdo_simd_safelen(
subroutine do_simd_safelen()
  ! CHECK:      omp.wsloop
  ! CHECK-NOT:  safelen({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: safelen({{.*}})
  !$omp do simd safelen(4)
    do index_ = 1, 10
    end do
  !$omp end do simd
end subroutine do_simd_safelen

! CHECK-LABEL: func.func @_QPdo_simd_simdlen(
subroutine do_simd_simdlen()
  ! CHECK:      omp.wsloop
  ! CHECK-NOT:  simdlen({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: simdlen({{.*}})
  !$omp do simd simdlen(4)
    do index_ = 1, 10
    end do
  !$omp end do simd
end subroutine do_simd_simdlen
