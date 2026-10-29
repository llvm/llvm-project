! This test checks lowering of OpenMP DO SIMD composite constructs.

! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir %s -o - | FileCheck %s --check-prefix=COPYBACK

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

! Verify that the simd reduction var references the wsloop's reduction
! block arg (not the original array), ensuring proper chaining of
! per-SIMD-lane results into the wsloop's thread-private reduction copy.
! CHECK-LABEL: {{.*}}do_simd_array_reduction{{.*}}
subroutine do_simd_array_reduction()
  integer :: a(100)
  a = 0
  ! CHECK:      omp.wsloop
  ! CHECK-SAME: reduction(byref @[[ADD_RED_SYM:.*]] %{{.*}} -> %[[ADD_RED_OUTER:.*]] : !fir.ref<!fir.box<!fir.array<100xi32>>>)
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: reduction(byref @[[ADD_RED_SYM]] %[[ADD_RED_OUTER]] -> %[[ADD_RED_INNER:.*]] : !fir.ref<!fir.box<!fir.array<100xi32>>>)
  ! CHECK-NEXT: omp.loop_nest
  !$omp do simd reduction(+:a)
    do index_ = 1, 10
      a = a + index_
    end do
  !$omp end do simd
end subroutine do_simd_array_reduction

! CHECK-LABEL: func.func @_QPdo_simd_private(
subroutine do_simd_private()
  integer, allocatable :: tmp
  ! CHECK:      %[[I_DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFdo_simd_privateEi")
  ! CHECK:      omp.wsloop
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: linear(@{{.*}} %[[I_DECL]]#0 -> %[[I_ARG:[^ ]+]] :
  ! CHECK-SAME: private(@[[PRIV_BOX_SYM:.*]] %{{.*}} -> %[[PRIV_BOX:.*]] : !fir.ref<!fir.box<!fir.heap<i32>>>)
  ! CHECK-NEXT: omp.loop_nest (%[[IVAR:.*]]) : i32
  !$omp do simd private(tmp)
  do i=1, 10
  ! CHECK:      %[[I_PRIV:.*]]:2 = hlfir.declare %[[I_ARG]]
  ! CHECK:      %[[PRIV_BOX_DECL:.*]]:2 = hlfir.declare %[[PRIV_BOX]]
  ! CHECK:      hlfir.assign %[[IVAR]] to %[[I_PRIV]]#0
  ! CHECK:      %[[PRIV_BOX_LOAD:.*]] = fir.load %[[PRIV_BOX_DECL]]
  ! CHECK:      hlfir.assign %{{.*}} to %[[PRIV_BOX_DECL]]#0
  ! CHECK:      omp.yield
    tmp = tmp + 1
  end do
end subroutine do_simd_private

! CHECK-LABEL: func.func @_QPdo_simd_lastprivate_firstprivate(
! Ensure each lastprivatised variable has exactly one lastprivate copyback.
! COPYBACK-LABEL: func.func @_QPdo_simd_lastprivate_firstprivate(
! COPYBACK: %[[ORIGINAL_A:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFdo_simd_lastprivate_firstprivateEa")
! COPYBACK-NOT: return
! COPYBACK-COUNT-1: hlfir.assign %{{.*}} to %[[ORIGINAL_A]]#0
! COPYBACK-NOT: hlfir.assign %{{.*}} to %[[ORIGINAL_A]]#0
! COPYBACK: return
subroutine do_simd_lastprivate_firstprivate()
  integer :: a
  ! CHECK: %[[ORIGINAL_A:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFdo_simd_lastprivate_firstprivateEa")
  ! CHECK:      omp.wsloop
  ! CHECK-SAME: private(@[[FIRSTPRIVATE_A_SYM:.*]] %[[ORIGINAL_A]]#0 -> %[[FIRSTPRIVATE_A:.*]] : !fir.ref<i32>)
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: linear(@{{.*}}#0 -> %{{[^ ]+}} : !fir.ref<i32> = %{{[^:]*}} : i32)
  ! CHECK-NOT: private(
  ! CHECK-NEXT: omp.loop_nest
  !$omp do simd lastprivate(a) firstprivate(a)
  do i = 1, 10
    ! CHECK: %[[FIRSTPRIVATE_A_DECL:.*]]:2 = hlfir.declare %[[FIRSTPRIVATE_A]]
    ! CHECK: %[[A_VALUE:.*]] = fir.load %[[FIRSTPRIVATE_A_DECL]]#0
    ! CHECK: %[[A_UPDATED:.*]] = arith.addi %[[A_VALUE]], %{{.*}}
    ! CHECK: hlfir.assign %[[A_UPDATED]] to %[[FIRSTPRIVATE_A_DECL]]#0
    a = a + 1
    ! CHECK: fir.if %{{.*}} {
    ! CHECK: %[[A_LAST:.*]] = fir.load %[[FIRSTPRIVATE_A_DECL]]#0
    ! CHECK: hlfir.assign %[[A_LAST]] to %[[ORIGINAL_A]]#0
  end do
  !$omp end do simd
end subroutine do_simd_lastprivate_firstprivate

! Firstprivate initialization must finish before any lastprivate copy-back,
! even with nowait. Hence the wsloop which owns the storage and requests
! the barrier. The simd reuses that storage and must not request another
! barrier.
! CHECK-LABEL: func.func @_QPdo_simd_first_lastprivate_barrier(
subroutine do_simd_first_lastprivate_barrier(a)
  integer :: a, i
  ! CHECK: omp.wsloop nowait private(@{{.*}}firstprivate{{.*}}) private_barrier {
  ! CHECK-NEXT: omp.simd
  ! CHECK-NOT: {{[ ]}}private_barrier
  ! CHECK-NEXT: omp.loop_nest
  !$omp do simd firstprivate(a) lastprivate(a)
  do i = 1, 10
    a = a + i
  end do
  !$omp end do simd nowait
end subroutine do_simd_first_lastprivate_barrier

! Reusing storage must preserve SIMD-only privates and collapsed loop bounds.
! CHECK-LABEL: func.func @_QPdo_simd_first_lastprivate_collapse(
! Ensure exactly one lastprivate copyback is produced
! COPYBACK-LABEL: func.func @_QPdo_simd_first_lastprivate_collapse(
! COPYBACK: %[[COLLAPSE_ORIGINAL_A:.*]]:2 = hlfir.declare %arg0
! COPYBACK-NOT: return
! COPYBACK-COUNT-1: hlfir.assign %{{.*}} to %[[COLLAPSE_ORIGINAL_A]]#0
! COPYBACK-NOT: hlfir.assign %{{.*}} to %[[COLLAPSE_ORIGINAL_A]]#0
! COPYBACK: return
subroutine do_simd_first_lastprivate_collapse(a)
  integer :: a, b, i, j
  ! CHECK: %[[COLLAPSE_ORIGINAL_A:.*]]:2 = hlfir.declare %arg0
  ! CHECK: omp.parallel {
  ! CHECK: omp.wsloop private(@{{.*}}firstprivate{{.*}} %[[COLLAPSE_ORIGINAL_A]]#0 -> %[[COLLAPSE_A:.*]] : !fir.ref<i32>)
  ! CHECK-SAME private_barrier
  ! CHECK-NEXT: omp.simd private(
  ! CHECK-SAME: -> %[[COLLAPSE_B:[^, ]+]],
  ! CHECK-NEXT: omp.loop_nest (%{{.*}}, %{{.*}}) : i32
  !$omp parallel do simd collapse(2) firstprivate(a) lastprivate(a) private(b)
  do i = 1, 2
    do j = 1, 2
      ! CHECK: %[[COLLAPSE_A_DECL:.*]]:2 = hlfir.declare %[[COLLAPSE_A]]
      ! CHECK: %[[COLLAPSE_B_DECL:.*]]:2 = hlfir.declare %[[COLLAPSE_B]]
      ! CHECK: %[[COLLAPSE_A_VALUE:.*]] = fir.load %[[COLLAPSE_A_DECL]]#0
      ! CHECK: hlfir.assign %[[COLLAPSE_A_VALUE]] to %[[COLLAPSE_B_DECL]]#0
      b = a
      a = b + i + j
      ! CHECK: fir.if %{{.*}} {
      ! CHECK: %[[COLLAPSE_A_LAST:.*]] = fir.load %[[COLLAPSE_A_DECL]]#0
      ! CHECK: hlfir.assign %[[COLLAPSE_A_LAST]] to %[[COLLAPSE_ORIGINAL_A]]#0
    end do
  end do
  !$omp end parallel do simd
end subroutine
