! Tests LOCAL_INIT + REDUCE together through the device OpenMP conversion
! pipeline. Verifies that LOCAL_INIT's per-iteration copy is inlined at the
! top of the omp.loop_nest body and the reduction clause is on omp.wsloop,
! within the full target teams distribute parallel do structure.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=device %s -o - \
! RUN:   | FileCheck %s

subroutine test_local_init_reduce_device(n)
  implicit none
  integer, intent(in) :: n
  real :: tmp, total
  integer :: i

  tmp = 1.0
  total = 0.0

  do concurrent (i = 1:n) local_init(tmp) reduce(+:total)
    tmp = tmp + real(i)
    total = total + tmp
  end do
end subroutine

! Verify: omp.private for LOCAL_INIT (no copy_region)
! CHECK:      omp.private {type = private} @[[PRIVATIZER:.*tmp.*\.omp]] : f32

! Verify: full device nesting with reduction on teams+wsloop, private on parallel
! CHECK:      omp.target
! CHECK:        omp.teams reduction(@{{.*}} %{{.*}} -> %{{.*}} : !fir.ref<f32>) {
! CHECK:          omp.parallel private(@[[PRIVATIZER]] %[[ORIG:.*]] -> %[[PRIV:.*]] : !fir.ref<f32>) {
! CHECK:            omp.distribute {
! CHECK:              omp.wsloop reduction(@{{.*}} %{{.*}} -> %{{.*}} : !fir.ref<f32>) {
! CHECK:                omp.loop_nest

! Verify: per-iteration copy at top of loop_nest body
! CHECK:                  %[[VAL:.*]] = fir.load %[[ORIG]] : !fir.ref<f32>
! CHECK-NEXT:             hlfir.assign %[[VAL]] to %[[PRIV]] : f32, !fir.ref<f32>
