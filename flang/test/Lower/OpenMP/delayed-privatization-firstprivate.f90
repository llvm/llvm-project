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
! CHECK-SAME: @[[PRIVATIZER_SYM:.*]] : [[TYPE:!fir.shadow<!fir.ref<i32>, !fir.ref<i32>, allocatable : false>]] alloc {
! CHECK-NEXT: ^bb0(%[[PRIV_ARG:.*]]: [[TYPE]]):

! CHECK-NEXT:  %[[BASE:.*]] = fir.extract_value %[[PRIV_ARG]], [0 : index] : ([[TYPE]]) -> !fir.ref<i32>
! CHECK-NEXT:  %[[FIR_BASE:.*]] = fir.extract_value %[[PRIV_ARG]], [1 : index] : ([[TYPE]]) -> !fir.ref<i32>

! CHECK-NEXT:   %[[PRIV_ALLOC:.*]] = fir.alloca i32 {bindc_name = "var1", pinned, uniq_name = "_QFdelayed_privatization_firstprivateEvar1"}
! CHECK-NEXT:   %[[PRIV_DECL:.*]]:2 = hlfir.declare %[[PRIV_ALLOC]] {uniq_name = "_QFdelayed_privatization_firstprivateEvar1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

! CHECK-NEXT:  %[[YIELD_VAL:.*]] = fir.undefined [[TYPE]]
! CHECK-NEXT:  %[[YIELD_VAL_2:.*]] = fir.insert_value %[[YIELD_VAL]], %[[PRIV_DECL]]#0, [0 : index] : ([[TYPE]], !fir.ref<i32>) -> [[TYPE]]
! CHECK-NEXT:  %[[YIELD_VAL_3:.*]] = fir.insert_value %[[YIELD_VAL_2]], %[[PRIV_DECL]]#1, [1 : index] : ([[TYPE]], !fir.ref<i32>) -> [[TYPE]]

! CHECK-NEXT:   omp.yield(%[[YIELD_VAL_3]] : [[TYPE]])

! CHECK: } copy {
! CHECK: ^bb0(%[[PRIV_ORIG_ARG:.*]]: [[TYPE]], %[[PRIV_PRIV_ARG:.*]]: [[TYPE]]):

! CHECK-NEXT:  %[[ORIG_BASE:.*]] = fir.extract_value %[[PRIV_ORIG_ARG]], [0 : index] : ([[TYPE]]) -> !fir.ref<i32>
! CHECK-NEXT:  %[[ORIG_FIR_BASE:.*]] = fir.extract_value %[[PRIV_ORIG_ARG]], [1 : index] : ([[TYPE]]) -> !fir.ref<i32>

! CHECK-NEXT:  %[[PRIV_BASE:.*]] = fir.extract_value %[[PRIV_PRIV_ARG]], [0 : index] : ([[TYPE]]) -> !fir.ref<i32>
! CHECK-NEXT:  %[[PRIV_FIR_BASE:.*]] = fir.extract_value %[[PRIV_PRIV_ARG]], [1 : index] : ([[TYPE]]) -> !fir.ref<i32>

! CHECK-NEXT:  %[[ORIG_VAL:.*]] = fir.load %[[ORIG_BASE]] : !fir.ref<i32>
! CHECK-NEXT:  hlfir.assign %[[ORIG_VAL]] to %[[PRIV_BASE]] temporary_lhs : i32, !fir.ref<i32>
! CHECK-NEXT:  omp.yield(%[[PRIV_PRIV_ARG]] : [[TYPE]])

! CHECK-LABEL: @_QPdelayed_privatization_firstprivate
! CHECK: omp.parallel private(@[[PRIVATIZER_SYM]] %{{.*}} -> %{{.*}} : [[TYPE]]) {
! CHECK: omp.terminator
