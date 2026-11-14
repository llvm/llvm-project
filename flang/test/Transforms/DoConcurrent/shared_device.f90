! Tests that the SHARED locality specifier works correctly through the
! do concurrent to OpenMP device conversion pipeline.
!
! On the device path, the shared variable must be mapped to the device via
! omp.map.info and then used as a live-in inside the target region. It should
! NOT appear on any private clause.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=device %s -o - \
! RUN:   | FileCheck %s

subroutine test_shared_device(n)
  implicit none
  integer, intent(in) :: n
  real :: shared_val
  real :: results(10)
  integer :: i

  shared_val = 42.0

  do concurrent (i = 1:10) shared(shared_val)
    results(i) = shared_val + real(i)
  end do
end subroutine

! Verify: fir.local {type = shared} exists at module level
! CHECK: fir.local {type = shared} @{{.*}}shared_val{{.*}} : f32

! Verify: shared_val is mapped to the device
! CHECK: omp.map.info var_ptr(%{{.*}} : !fir.ref<f32>, f32) map_clauses(implicit)

! Verify: omp.target + omp.teams + omp.parallel structure with NO private
! clause for shared_val
! CHECK:      omp.target
! CHECK:        omp.teams {
! CHECK:          omp.parallel {
! CHECK:            omp.distribute {
! CHECK:              omp.wsloop {
! CHECK:                omp.loop_nest
