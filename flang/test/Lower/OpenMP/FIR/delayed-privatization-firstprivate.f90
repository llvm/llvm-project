! Test delayed privatization for the `private` clause.

! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir -fopenmp -mmlir \
! RUN:   --openmp-enable-delayed-privatization -o - %s 2>&1 | FileCheck %s
! RUN: bbc -emit-fir -hlfir=false -fopenmp --openmp-enable-delayed-privatization \
! RUN:   -o - %s 2>&1 | FileCheck %s

subroutine delayed_privatization_firstprivate
  implicit none
  integer :: var1

!$OMP PARALLEL FIRSTPRIVATE(var1)
  var1 = 10
!$OMP END PARALLEL
end subroutine

! CHECK-LABEL: omp.private {type = firstprivate}
! CHECK-SAME: @[[VAR1_PRIVATIZER_SYM:.*]] : !fir.ref<i32> alloc {
! CHECK-NEXT: ^bb0(%[[PRIV_ARG:.*]]: !fir.ref<i32>):
! CHECK-NEXT:   %[[PRIV_ALLOC:.*]] = fir.alloca i32 {bindc_name = "var1", pinned, uniq_name = "_QFdelayed_privatization_firstprivateEvar1"}
! CHECK-NEXT:   omp.yield(%[[PRIV_ALLOC]] : !fir.ref<i32>)
! CHECK: } copy {
! CHECK: ^bb0(%[[PRIV_ORIG_ARG:.*]]: !fir.ref<i32>, %[[PRIV_PRIV_ARG:.*]]: !fir.ref<i32>):
! CHECK:    %[[ORIG_VAL:.*]] = fir.load %[[PRIV_ORIG_ARG]] : !fir.ref<i32>
! CHECK:    fir.store %[[ORIG_VAL]] to %[[PRIV_PRIV_ARG]] : !fir.ref<i32>
! CHECK:    omp.yield(%[[PRIV_PRIV_ARG]] : !fir.ref<i32>)
! CHECK: }

! CHECK-LABEL: @_QPdelayed_privatization_firstprivate
! CHECK: omp.parallel private(@[[VAR1_PRIVATIZER_SYM]] %{{.*}} -> %{{.*}} : !fir.ref<i32>) {
! CHECK: omp.terminator

