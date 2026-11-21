! Tests that SHARED and LOCAL_INIT work correctly together in the same
! do concurrent loop through the host OpenMP conversion pipeline.
!
! The SHARED variable should be a live-in (no private clause), while the
! LOCAL_INIT variable should be privatized with per-iteration copy inlined
! at the top of the omp.loop_nest body.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=host %s -o - \
! RUN:   | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=host %s -o - \
! RUN:   | FileCheck %s

subroutine test_shared_local_init(a, n)
  implicit none
  integer, intent(in) :: n
  real, intent(inout) :: a(n)
  real :: shared_val
  real :: tmp
  integer :: i

  shared_val = 42.0
  tmp = 1.0

  do concurrent (i = 1:n) shared(shared_val) local_init(tmp)
    tmp = tmp + shared_val + real(i)
    a(i) = tmp
  end do
end subroutine

! Verify: both fir.local ops exist at module level
! CHECK-DAG: fir.local {type = shared} @{{.*}}shared_val{{.*}} : f32
! CHECK-DAG: fir.local {type = local_init} @{{.*}}tmp{{.*}} : f32

! Verify: omp.private exists for LOCAL_INIT (tmp), NOT for SHARED (shared_val)
! CHECK: omp.private {type = private} @[[TMP_PRIV:.*tmp.*\.omp]] : f32

! Verify: omp.wsloop has private clause only for tmp, not shared_val
! CHECK:      omp.parallel {
! CHECK:        omp.wsloop private(@[[TMP_PRIV]] %{{.*}} -> %{{.*}} : !fir.ref<f32>)

! Verify: per-iteration copy for LOCAL_INIT at top of loop_nest
! CHECK:          omp.loop_nest
! CHECK:            fir.load
! CHECK-NEXT:       hlfir.assign
