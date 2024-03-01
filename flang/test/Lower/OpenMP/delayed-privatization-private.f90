! Test delayed privatization for the `private` clause.

! RUN: bbc -emit-hlfir -fopenmp --openmp-enable-delayed-privatization -o - %s 2>&1 | FileCheck %s

subroutine delayed_privatization_private
  implicit none
  integer :: var1

!$omp parallel private(var1)
  var1 = 10
!$omp end parallel
end subroutine

! CHECK-LABEL: omp.private {type = private}
! CHECK-SAME: @[[PRIVATIZER_SYM:.*]] : [[TYPE:!fir.shadow<!fir.ref<i32>, !fir.ref<i32>, allocatable : false>]] alloc {
! CHECK-NEXT: ^bb0(%[[PRIV_ARG:.*]]: [[TYPE]]):

! CHECK-NEXT:  %[[BASE:.*]] = fir.extract_value %[[PRIV_ARG]], [0 : index] : ([[TYPE]]) -> !fir.ref<i32>
! CHECK-NEXT:  %[[FIR_BASE:.*]] = fir.extract_value %[[PRIV_ARG]], [1 : index] : ([[TYPE]]) -> !fir.ref<i32>

! CHECK-NEXT:   %[[PRIV_ALLOC:.*]] = fir.alloca i32 {bindc_name = "var1", pinned, uniq_name = "_QFdelayed_privatization_privateEvar1"}
! CHECK-NEXT:   %[[PRIV_DECL:.*]]:2 = hlfir.declare %[[PRIV_ALLOC]] {uniq_name = "_QFdelayed_privatization_privateEvar1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

! CHECK-NEXT:  %[[YIELD_VAL:.*]] = fir.undefined [[TYPE]]
! CHECK-NEXT:  %[[YIELD_VAL_2:.*]] = fir.insert_value %[[YIELD_VAL]], %[[PRIV_DECL]]#0, [0 : index] : ([[TYPE]], !fir.ref<i32>) -> [[TYPE]]
! CHECK-NEXT:  %[[YIELD_VAL_3:.*]] = fir.insert_value %[[YIELD_VAL_2]], %[[PRIV_DECL]]#1, [1 : index] : ([[TYPE]], !fir.ref<i32>) -> [[TYPE]]

! CHECK-NEXT:   omp.yield(%[[YIELD_VAL_3]] : [[TYPE]])
! CHECK-NOT: } copy {

! CHECK-LABEL: @_QPdelayed_privatization_private
! CHECK: %[[ORIG_ALLOC:.*]] = fir.alloca i32 {bindc_name = "var1", uniq_name = "_QFdelayed_privatization_privateEvar1"}
! CHECK: %[[ORIG_DECL:.*]]:2 = hlfir.declare %[[ORIG_ALLOC]] {uniq_name = "_QFdelayed_privatization_privateEvar1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

! CHECK-NEXT:  %[[ORIG_VAL:.*]] = fir.undefined [[TYPE]]
! CHECK-NEXT:  %[[ORIG_VAL_2:.*]] = fir.insert_value %[[ORIG_VAL]], %[[ORIG_DECL]]#0, [0 : index] : ([[TYPE]], !fir.ref<i32>) -> [[TYPE]]
! CHECK-NEXT:  %[[ORIG_VAL_3:.*]] = fir.insert_value %[[ORIG_VAL_2]], %[[ORIG_DECL]]#1, [1 : index] : ([[TYPE]], !fir.ref<i32>) -> [[TYPE]]

! CHECK: omp.parallel private(@[[PRIVATIZER_SYM]] %[[ORIG_VAL_3]] -> %[[PAR_ARG:.*]] : [[TYPE]]) {
! CHECK: %[[PAR_ARG_FIR_BASE:.*]] = fir.extract_value %[[PAR_ARG]], [1 : index] : ([[TYPE]]) -> !fir.ref<i32>
! CHECK: %[[PAR_ARG_DECL:.*]]:2 = hlfir.declare %[[PAR_ARG_FIR_BASE]] {uniq_name = "_QFdelayed_privatization_privateEvar1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: hlfir.assign %{{.*}} to %[[PAR_ARG_DECL]]#0 : i32, !fir.ref<i32>
! CHECK: omp.terminator
