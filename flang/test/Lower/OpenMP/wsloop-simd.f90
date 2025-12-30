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

! CHECK-LABEL: func.func @_QPdo_simd_private(
subroutine do_simd_private()
  integer, allocatable :: tmp
  ! CHECK:      omp.wsloop
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: private(@[[PRIV_BOX_SYM:.*]] %{{.*}} -> %[[PRIV_BOX:.*]], @[[PRIV_IVAR_SYM:.*]] %{{.*}} -> %[[PRIV_IVAR:.*]] : !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<i32>)
  ! CHECK-NEXT: omp.loop_nest (%[[IVAR:.*]]) : i32
  !$omp do simd private(tmp)
  do i=1, 10
  ! CHECK:      %[[PRIV_BOX_DECL:.*]]:2 = hlfir.declare %[[PRIV_BOX]]
  ! CHECK:      %[[PRIV_IVAR_DECL:.*]]:2 = hlfir.declare %[[PRIV_IVAR]]
  ! CHECK:      hlfir.assign %[[IVAR]] to %[[PRIV_IVAR_DECL]]#0
  ! CHECK:      %[[PRIV_BOX_LOAD:.*]] = fir.load %[[PRIV_BOX_DECL]]
  ! CHECK:      hlfir.assign %{{.*}} to %[[PRIV_BOX_DECL]]#0
  ! CHECK:      omp.yield
    tmp = tmp + 1
  end do
end subroutine do_simd_private

! CHECK-LABEL: func.func @_QPdo_simd_lastprivate_firstprivate(
subroutine do_simd_lastprivate_firstprivate()
  integer :: a
  ! CHECK:      omp.wsloop
  ! CHECK-SAME: private(@[[FIRSTPRIVATE_A_SYM:.*]] %{{.*}} -> %[[FIRSTPRIVATE_A:.*]] : !fir.ref<i32>)
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: private(@[[PRIVATE_A_SYM:.*]] %{{.*}} -> %[[PRIVATE_A:.*]], @[[PRIVATE_I_SYM:.*]] %{{.*}} -> %[[PRIVATE_I:.*]] : !fir.ref<i32>, !fir.ref<i32>)
  !$omp do simd lastprivate(a) firstprivate(a)
  do i = 1, 10
    ! CHECK: %[[FIRSTPRIVATE_A_DECL:.*]]:2 = hlfir.declare %[[FIRSTPRIVATE_A]]
    ! CHECK: %[[PRIVATE_A_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_A]]
    ! CHECK: %[[PRIVATE_I_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_I]]
    a = a + 1
  end do
  !$omp end do simd
end subroutine do_simd_lastprivate_firstprivate
