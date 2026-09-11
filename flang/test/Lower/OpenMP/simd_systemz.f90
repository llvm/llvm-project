! Tests for 2.9.3.1 Simd and target dependent alignment for SystemZ
! REQUIRES: systemz-registered-target
! RUN: %flang_fc1 -triple s390x-unknown-linux-gnu -emit-hlfir -fopenmp %s -o - | FileCheck %s
subroutine simdloop_aligned_cptr(A)
    use iso_c_binding
    integer :: i
    type (c_ptr) :: A
  !CHECK: omp.simd aligned(
  !CHECK-SAME: -> 64 : i64)
    !$OMP SIMD ALIGNED(A)
    do i = 1, 10
      call c_test_call(A)
    end do
    !$OMP END SIMD
end subroutine

subroutine simdloop_aligned_cptr_8(A)
    use iso_c_binding
    integer :: i
    type (c_ptr) :: A
  !CHECK: omp.simd aligned(
  !CHECK-SAME: -> 8 : i64)
    !$OMP SIMD ALIGNED(A:8)
    do i = 1, 10
      call c_test_call(A)
    end do
    !$OMP END SIMD
end subroutine
