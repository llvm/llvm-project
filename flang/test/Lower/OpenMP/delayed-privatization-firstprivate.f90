! Test delayed privatization for the `firstprivate` clause.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --openmp-enable-delayed-privatization \
! RUN:   -o - %s 2>&1 | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp --openmp-enable-delayed-privatization -o - %s 2>&1 \
! RUN:   | FileCheck %s

subroutine delayed_privatization_firstprivate
  implicit none
  integer :: var1

!$omp parallel firstprivate(var1)
  var1 = 10
!$omp end parallel
end subroutine

! CHECK-LABEL: omp.private {type = firstprivate}
! CHECK-SAME: @[[VAR1_PRIVATIZER_SYM:.*]] : i32 copy {
! CHECK: ^bb0(%[[PRIV_ORIG_ARG:.*]]: !fir.ref<i32>, %[[PRIV_PRIV_ARG:.*]]: !fir.ref<i32>):
! CHECK:    %[[ORIG_VAL:.*]] = fir.load %[[PRIV_ORIG_ARG]] : !fir.ref<i32>
! CHECK:    hlfir.assign %[[ORIG_VAL]] to %[[PRIV_PRIV_ARG]] : i32, !fir.ref<i32>
! CHECK:    omp.yield(%[[PRIV_PRIV_ARG]] : !fir.ref<i32>)
! CHECK: }

! CHECK-LABEL: @_QPdelayed_privatization_firstprivate
! CHECK: omp.parallel private(@[[VAR1_PRIVATIZER_SYM]] %{{.*}} -> %{{.*}} : !fir.ref<i32>) {
! CHECK: omp.terminator
