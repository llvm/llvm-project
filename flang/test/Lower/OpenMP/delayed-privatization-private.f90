! Test delayed privatization for the `private` clause.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --openmp-enable-delayed-privatization \
! RUN:   -o - %s 2>&1 | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp --openmp-enable-delayed-privatization -o - %s 2>&1 \
! RUN:   | FileCheck %s

subroutine delayed_privatization_private
  implicit none
  integer :: var1

!$omp parallel private(var1)
  var1 = 10
!$omp end parallel
end subroutine

! CHECK-LABEL: omp.private {type = private}
! CHECK-SAME: @[[PRIVATIZER_SYM:.*]] : !fir.ref<i32> alloc {
! CHECK-NEXT: ^bb0(%[[PRIV_ARG:.*]]: !fir.ref<i32>):
! CHECK-NEXT:   %[[PRIV_ALLOC:.*]] = fir.alloca i32 {bindc_name = "var1", pinned, uniq_name = "_QFdelayed_privatization_privateEvar1"}
! CHECK-NEXT:   %[[PRIV_DECL:.*]]:2 = hlfir.declare %[[PRIV_ALLOC]] {uniq_name = "_QFdelayed_privatization_privateEvar1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-NEXT:   omp.yield(%[[PRIV_DECL]]#0 : !fir.ref<i32>)
! CHECK-NOT: } copy {

! CHECK-LABEL: @_QPdelayed_privatization_private
! CHECK: %[[ORIG_ALLOC:.*]] = fir.alloca i32 {bindc_name = "var1", uniq_name = "_QFdelayed_privatization_privateEvar1"}
! CHECK: %[[ORIG_DECL:.*]]:2 = hlfir.declare %[[ORIG_ALLOC]] {uniq_name = "_QFdelayed_privatization_privateEvar1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: omp.parallel private(@[[PRIVATIZER_SYM]] %[[ORIG_DECL]]#0 -> %[[PAR_ARG:.*]] : !fir.ref<i32>) {
! CHECK: %[[PAR_ARG_DECL:.*]]:2 = hlfir.declare %[[PAR_ARG]] {uniq_name = "_QFdelayed_privatization_privateEvar1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: hlfir.assign %{{.*}} to %[[PAR_ARG_DECL]]#0 : i32, !fir.ref<i32>
! CHECK: omp.terminator
