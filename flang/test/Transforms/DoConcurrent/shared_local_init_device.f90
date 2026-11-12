! Tests SHARED + LOCAL_INIT together through the device OpenMP conversion
! pipeline. Verifies that the SHARED variable is mapped to the device and
! used as a live-in (no private clause), while LOCAL_INIT is privatized
! with per-iteration copy inlined in the loop body.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=device %s -o - \
! RUN:   | FileCheck %s

subroutine test_shared_local_init_device(n)
  implicit none
  integer, intent(in) :: n
  real :: shared_val
  real :: tmp
  real :: results(10)
  integer :: i

  shared_val = 42.0
  tmp = 1.0

  do concurrent (i = 1:10) shared(shared_val) local_init(tmp)
    tmp = tmp + shared_val + real(i)
    results(i) = tmp
  end do
end subroutine

! Verify: fir.local ops for both SHARED and LOCAL_INIT
! CHECK-DAG: fir.local {type = shared} @{{.*}}shared_val{{.*}} : f32
! CHECK-DAG: fir.local {type = local_init} @{{.*}}tmp{{.*}} : f32

! Verify: omp.private exists for LOCAL_INIT only (not for SHARED)
! CHECK: omp.private {type = private} @[[TMP_PRIV:.*tmp.*\.omp]] : f32

! Verify: device structure with private for LOCAL_INIT on parallel
! CHECK:      omp.target
! CHECK:        omp.teams {
! CHECK:          omp.parallel private(@[[TMP_PRIV]] %{{.*}} -> %{{.*}} : !fir.ref<f32>) {
! CHECK:            omp.distribute {
! CHECK:              omp.wsloop {
! CHECK:                omp.loop_nest
