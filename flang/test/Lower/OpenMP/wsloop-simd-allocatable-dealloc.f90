! RUN: bbc -fopenmp -emit-hlfir  %s -o - | FileCheck %s --implicit-check-not=fir.freemem
! RUN: %flang_fc1 -fopenmp -emit-hlfir  %s -o - | FileCheck %s --implicit-check-not=fir.freemem

! Each allocatable has one cleanup recipe. The SIMD-only private b retains
! its cleanup, while a uses the wsloop's firstprivate storage and cleanup.
! No inline frees should be emitted by insertDeallocs for delayed privates.
! CHECK-LABEL: omp.private {type = private}
! CHECK-SAME: @[[B_RECIPE:.*]] : !fir.box<!fir.heap<i32>> init {
! CHECK: } dealloc {
! CHECK-NEXT: ^bb0(%[[B:.*]]: !fir.ref<!fir.box<!fir.heap<i32>>>):
! CHECK-NEXT: %[[B_BOX:.*]] = fir.load %[[B]]
! CHECK-NEXT: %[[B_ADDR:.*]] = fir.box_addr %[[B_BOX]]
! CHECK: fir.if %{{.*}} {
! CHECK-NEXT: fir.freemem %[[B_ADDR]] : !fir.heap<i32>
! CHECK-NEXT: }
! CHECK-NEXT: omp.yield
! CHECK-NEXT: }
! CHECK-LABEL: omp.private {type = firstprivate}
! CHECK-SAME: @[[A_RECIPE:.*]] : !fir.box<!fir.heap<i32>> init {
! CHECK: } dealloc {
! CHECK-NEXT: ^bb0(%[[A:.*]]: !fir.ref<!fir.box<!fir.heap<i32>>>):
! CHECK-NEXT: %[[A_BOX:.*]] = fir.load %[[A]]
! CHECK-NEXT: %[[A_ADDR:.*]] = fir.box_addr %[[A_BOX]]
! CHECK: fir.if %{{.*}} {
! CHECK-NEXT: fir.freemem %[[A_ADDR]] : !fir.heap<i32>
! CHECK-NEXT: }
! CHECK-NEXT: omp.yield
! CHECK-NEXT: }

! CHECK-LABEL: func.func @_QPreused_cleanup(
subroutine reused_cleanup(a, b)
  integer, allocatable :: a, b
  integer :: i
  ! CHECK: omp.wsloop private(@[[A_RECIPE]] %{{.*}} -> %[[PRIVATE_A:.*]] : !fir.ref<!fir.box<!fir.heap<i32>>>) private_barrier {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: private(@[[B_RECIPE]] %{{.*}} -> %[[PRIVATE_B:.*]] : !fir.ref<!fir.box<!fir.heap<i32>>>) {
  ! CHECK-NEXT: omp.loop_nest
  ! CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_A]]
  ! CHECK: %[[B_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_B]]
  !$omp do simd firstprivate(a) lastprivate(a) private(b)
  do i = 1, 10
    b = i
    a = b
  end do
  !$omp end do simd
  ! CHECK: return
end subroutine
