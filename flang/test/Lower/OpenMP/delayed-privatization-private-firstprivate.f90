! Test delayed privatization for both `private` and `firstprivate` clauses.

! RUN: bbc -emit-hlfir -fopenmp --openmp-enable-delayed-privatization -o - %s 2>&1 | FileCheck %s

subroutine delayed_privatization_private_firstprivate
  implicit none
  integer :: var1
  integer :: var2

!$omp parallel private(var1) firstprivate(var2)
  var1 = 10
  var2 = var1 + var2
!$omp end parallel
end subroutine

! CHECK-LABEL: omp.private {type = firstprivate}
! CHECK-SAME: @[[VAR2_PRIVATIZER_SYM:.*]] : [[TYPE:!fir.shadow<!fir.ref<i32>, !fir.ref<i32>, allocatable : false>]] alloc {
! CHECK: } copy {
! CHECK: }

! CHECK-LABEL: omp.private {type = private}
! CHECK-SAME: @[[VAR1_PRIVATIZER_SYM:.*]] : [[TYPE]] alloc {
! CHECK: }

! CHECK-LABEL: func.func @_QPdelayed_privatization_private_firstprivate() {
! CHECK:  %[[VAR1_ALLOC:.*]] = fir.alloca i32 {bindc_name = "var1"
! CHECK:  %[[VAR1_DECL:.*]]:2 = hlfir.declare %[[VAR1_ALLOC]]

! CHECK:  %[[VAR2_ALLOC:.*]] = fir.alloca i32 {bindc_name = "var2"
! CHECK:  %[[VAR2_DECL:.*]]:2 = hlfir.declare %[[VAR2_ALLOC]]

! CHECK:  %[[VAR1_VAL:.*]] = fir.undefined [[TYPE]]
! CHECK:  %[[VAR1_VAL_2:.*]] = fir.insert_value %[[VAR1_VAL]], %[[VAR1_DECL]]#0, [0 : index]
! CHECK:  %[[VAR1_VAL_3:.*]] = fir.insert_value %[[VAR1_VAL_2]], %[[VAR1_DECL]]#1, [1 : index]

! CHECK:  %[[VAR2_VAL:.*]] = fir.undefined [[TYPE]]
! CHECK:  %[[VAR2_VAL_2:.*]] = fir.insert_value %[[VAR2_VAL]], %[[VAR2_DECL]]#0, [0 : index]
! CHECK:  %[[VAR2_VAL_3:.*]] = fir.insert_value %[[VAR2_VAL_2]], %[[VAR2_DECL]]#1, [1 : index]

! CHECK:  omp.parallel private(
! CHECK-SAME: @[[VAR1_PRIVATIZER_SYM]] %[[VAR1_VAL_3]] -> %{{.*}} : [[TYPE]],
! CHECK-SAME: @[[VAR2_PRIVATIZER_SYM]] %[[VAR2_VAL_3]] -> %{{.*}} : [[TYPE]]) {
