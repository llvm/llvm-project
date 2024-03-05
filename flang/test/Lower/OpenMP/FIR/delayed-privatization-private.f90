! Test delayed privatization for the `private` clause.

! RUN: bbc -emit-fir -hlfir=false -fopenmp --openmp-enable-delayed-privatization -o - %s 2>&1 | FileCheck %s

subroutine delayed_privatization_private
  implicit none
  integer :: var1

!$OMP PARALLEL PRIVATE(var1)
  var1 = 10
!$OMP END PARALLEL

!$OMP PARALLEL PRIVATE(var1)
  var1 = 20
!$OMP END PARALLEL

end subroutine

! CHECK-LABEL: omp.private {type = private}
! CHECK-SAME: @[[PRIVATIZER_SYM:.*]] : !fir.ref<i32> alloc {
! CHECK-NEXT: ^bb0(%[[PRIV_ARG:.*]]: !fir.ref<i32>):
! CHECK-NEXT:   %[[PRIV_ALLOC:.*]] = fir.alloca i32 {bindc_name = "var1", pinned, uniq_name = "_QFdelayed_privatization_privateEvar1"}
! CHECK-NEXT:   omp.yield(%[[PRIV_ALLOC]] : !fir.ref<i32>)
! CHECK-NOT: } copy {

! CHECK-LABEL: @_QPdelayed_privatization_private
! CHECK: %[[ORIG_ALLOC:.*]] = fir.alloca i32 {bindc_name = "var1", uniq_name = "_QFdelayed_privatization_privateEvar1"}
! CHECK: omp.parallel private(@[[PRIVATIZER_SYM]] %[[ORIG_ALLOC]] -> %[[PAR_ARG:.*]] : !fir.ref<i32>) {
! CHECK: %[[C10:.*]] = arith.constant 10 : i32
! CHECK: fir.store %[[C10]] to %[[PAR_ARG]] : !fir.ref<i32>
! CHECK: omp.terminator

! Test that the same privatizer is used if the a variable with the same type and
! name was previously privatized.
! CHECK: omp.parallel private(@[[PRIVATIZER_SYM]] %[[ORIG_ALLOC]] -> %[[PAR_ARG:.*]] : !fir.ref<i32>) {
! CHECK: %[[C20:.*]] = arith.constant 20 : i32
! CHECK: fir.store %[[C20]] to %[[PAR_ARG]] : !fir.ref<i32>
! CHECK: omp.terminator
