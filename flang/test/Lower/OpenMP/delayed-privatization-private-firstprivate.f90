! Test delayed privatization for both `private` and `firstprivate` clauses.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --openmp-enable-delayed-privatization \
! RUN:   -o - %s 2>&1 | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp --openmp-enable-delayed-privatization -o - %s 2>&1 \
! RUN:   | FileCheck %s

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
! CHECK-SAME: @[[VAR2_PRIVATIZER_SYM:.*]] : !fir.ref<i32> alloc {
! CHECK: } copy {
! CHECK: }

! CHECK-LABEL: omp.private {type = private}
! CHECK-SAME: @[[VAR1_PRIVATIZER_SYM:.*]] : !fir.ref<i32> alloc {
! CHECK: }

! CHECK-LABEL: func.func @_QPdelayed_privatization_private_firstprivate() {
! CHECK:  %[[VAR1_ALLOC:.*]] = fir.alloca i32 {bindc_name = "var1"
! CHECK:  %[[VAR1_DECL:.*]]:2 = hlfir.declare %[[VAR1_ALLOC]]

! CHECK:  %[[VAR2_ALLOC:.*]] = fir.alloca i32 {bindc_name = "var2"
! CHECK:  %[[VAR2_DECL:.*]]:2 = hlfir.declare %[[VAR2_ALLOC]]

! CHECK:  omp.parallel private(
! CHECK-SAME: @[[VAR1_PRIVATIZER_SYM]] %[[VAR1_DECL]]#0 -> %{{[^,]+}}, 
! CHECK-SAME: @[[VAR2_PRIVATIZER_SYM]] %[[VAR2_DECL]]#0 -> %{{.*}} :
! CHECK-SAME: !fir.ref<i32>, !fir.ref<i32>) {
