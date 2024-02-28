! Test delayed privatization for the `firstprivate` clause.

! RUN: bbc -emit-hlfir -fopenmp --openmp-enable-delayed-privatization -o - %s 2>&1 | FileCheck %s

subroutine delayed_privatization_firstprivate
  implicit none
  integer :: var1

!$omp parallel firstprivate(var1)
  var1 = 10
!$omp end parallel
end subroutine

! CHECK-LABEL: omp.private {type = firstprivate}
! CHECK-SAME: @[[VAR1_PRIVATIZER_SYM:.*]] : !fir.ref<i32> alloc {
! CHECK-NEXT: ^bb0(%[[PRIV_ARG:.*]]: !fir.ref<i32>):
! CHECK-NEXT:   %[[PRIV_ALLOC:.*]] = fir.alloca i32 {bindc_name = "var1", pinned, uniq_name = "_QFdelayed_privatization_firstprivateEvar1"}
! CHECK-NEXT:   %[[PRIV_DECL:.*]]:2 = hlfir.declare %[[PRIV_ALLOC]] {uniq_name = "_QFdelayed_privatization_firstprivateEvar1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-NEXT:   omp.yield(%[[PRIV_DECL]]#0 : !fir.ref<i32>)
! CHECK: } copy {
! CHECK: ^bb0(%[[PRIV_ORIG_ARG:.*]]: !fir.ref<i32>, %[[PRIV_PRIV_ARG:.*]]: !fir.ref<i32>):
! CHECK:    %[[ORIG_VAL:.*]] = fir.load %[[PRIV_ORIG_ARG]] : !fir.ref<i32>
! CHECK:    hlfir.assign %[[ORIG_VAL]] to %[[PRIV_PRIV_ARG]] temporary_lhs : i32, !fir.ref<i32>
! CHECK:    omp.yield(%[[PRIV_PRIV_ARG]] : !fir.ref<i32>)
! CHECK: }

! CHECK-LABEL: @_QPdelayed_privatization_firstprivate
! CHECK: omp.parallel private(@[[VAR1_PRIVATIZER_SYM]] %{{.*}} -> %{{.*}} : !fir.ref<i32>) {
! CHECK: omp.terminator
