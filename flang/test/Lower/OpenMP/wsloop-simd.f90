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

! CHECK-LABEL: func.func @_QPdo_simd_reduction(
subroutine do_simd_reduction()
  integer :: sum
  sum = 0
  ! CHECK:      omp.wsloop
  ! CHECK-SAME: reduction(@[[RED_SYM:.*]] %{{.*}} -> %[[RED_OUTER:.*]] : !fir.ref<i32>)
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: reduction(@[[RED_SYM]] %[[RED_OUTER]] -> %[[RED_INNER:.*]] : !fir.ref<i32>)
  ! CHECK-NEXT: omp.loop_nest
  ! CHECK:      %[[RED_DECL:.*]]:2 = hlfir.declare %[[RED_INNER]]
  ! CHECK:      %[[RED:.*]] = fir.load %[[RED_DECL]]#0 : !fir.ref<i32>
  ! CHECK:      %[[RESULT:.*]] = arith.addi %[[RED]], %{{.*}} : i32
  ! CHECK:      hlfir.assign %[[RESULT]] to %[[RED_DECL]]#0 : i32, !fir.ref<i32>
  ! CHECK-NEXT: omp.yield
  !$omp do simd reduction(+:sum)
    do index_ = 1, 10
      sum = sum + 1
    end do
  !$omp end do simd
end subroutine do_simd_reduction
