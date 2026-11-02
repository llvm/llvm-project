! Tests end-to-end conversion of LOCAL_INIT with multi-dimensional do
! concurrent from Fortran source through the full pipeline.
!
! Verifies that per-iteration copy fires correctly in a collapsed
! multi-dimensional loop — each (i,j) pair should see a fresh copy of
! the LOCAL_INIT variable initialized from the original.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=host %s -o - \
! RUN:   | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=host %s -o - \
! RUN:   | FileCheck %s

subroutine test_multidim(a, n, m)
  implicit none
  integer, intent(in) :: n, m
  real, intent(inout) :: a(n, m)
  real :: tmp
  integer :: i, j

  tmp = 1.0

  do concurrent (i = 1:n, j = 1:m) local_init(tmp)
    tmp = tmp + real(i) + real(j)
    a(i, j) = tmp
  end do
end subroutine

! Verify: omp.private for LOCAL_INIT
! CHECK:      omp.private {type = private} @[[PRIVATIZER:.*tmp.*\.omp]] : f32

! Verify: collapsed loop with 2 induction variables
! CHECK:      omp.wsloop private(@[[PRIVATIZER]] %[[ORIG:.*]] -> %[[PRIV:.*]] : !fir.ref<f32>)
! CHECK:        omp.loop_nest (%{{.*}}, %{{.*}}) : index
! CHECK-SAME:     collapse(2)

! Verify: per-iteration copy at top of collapsed loop body
! CHECK:          %[[VAL:.*]] = fir.load %[[ORIG]] : !fir.ref<f32>
! CHECK-NEXT:     hlfir.assign %[[VAL]] to %[[PRIV]] : f32, !fir.ref<f32>
