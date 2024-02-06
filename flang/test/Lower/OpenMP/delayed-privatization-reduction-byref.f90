! Test that reductions and delayed privatization work properly togehter. Since
! both types of clauses add block arguments to the OpenMP region, we make sure
! that the block arguments are added in the proper order (reductions first and
! then delayed privatization.

! RUN: bbc -emit-hlfir -fopenmp --force-byref-reduction --openmp-enable-delayed-privatization -o - %s 2>&1 | FileCheck %s

subroutine red_and_delayed_private
    integer :: red
    integer :: prv

    red = 0
    prv = 10

    !$omp parallel reduction(+:red) private(prv)
    red = red + 1
    prv = 20
    !$omp end parallel
end subroutine

! CHECK-LABEL: omp.private {type = private}
! CHECK-SAME: @[[PRIVATIZER_SYM:.*]] : !fir.ref<i32> alloc {

! CHECK-LABEL: omp.reduction.declare
! CHECK-SAME: @[[REDUCTION_SYM:.*]] : !fir.ref<i32> init

! CHECK-LABEL: _QPred_and_delayed_private
! CHECK: omp.parallel
! CHECK-SAME: reduction(@[[REDUCTION_SYM]] %{{.*}} -> %arg0 : !fir.ref<i32>)
! CHECK-SAME: private(@[[PRIVATIZER_SYM]] %{{.*}} -> %arg1 : !fir.ref<i32>) {
