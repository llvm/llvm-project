! Tests end-to-end conversion of LOCAL_INIT locality specifier from Fortran
! source through the full pipeline (parsing -> lowering -> OpenMP conversion).
!
! LOCAL_INIT requires per-iteration initialization: at the start of each
! iteration, the private variable is re-initialized from the host variable.
! This is implemented by inlining the fir.local copy_region's operations at
! the top of the omp.loop_nest body. We do not place the copy_region on the
! omp.private op because the LLVM IR translation (copyFirstPrivateVars in
! OpenMPToLLVMIRTranslation.cpp) fires omp.private's copy_region once per
! thread before the worksharing loop, not once per iteration. Inlining
! directly into the loop body ensures every iteration re-initializes from
! the original.
!
! The test verifies:
!   1. omp.private has type=private (no copy_region)
!   2. Per-iteration copy (load from original, assign to private) appears
!      at the top of the omp.loop_nest body
!   3. The rest of the loop body follows the copy

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=host %s -o - \
! RUN:   | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=host %s -o - \
! RUN:   | FileCheck %s

subroutine test_local_init(a, n)
  implicit none
  integer, intent(in) :: n
  real, intent(inout) :: a(n)
  integer :: i
  real :: tmp

  tmp = 3.14

  do concurrent (i = 1:n) local_init(tmp)
    tmp = tmp + real(i)
    a(i) = tmp
  end do
end subroutine

! Verify: omp.private has type=private (no copy_region — same as LOCAL)
! CHECK:      omp.private {type = private} @[[PRIVATIZER:.*tmp.*\.omp]] : f32

! Verify: omp.parallel wrapping omp.wsloop with private clause
! CHECK:      omp.parallel {
! CHECK:        omp.wsloop private(@[[PRIVATIZER]] %[[ORIG:.*]] -> %[[PRIV:.*]] : !fir.ref<f32>)

! Verify: per-iteration copy inlined at the top of omp.loop_nest body.
! The copy loads from the original variable and assigns to the private.
! CHECK:          omp.loop_nest (%{{.*}}) {{.*}} {
! CHECK-NEXT:       %[[VAL:.*]] = fir.load %[[ORIG]] : !fir.ref<f32>
! CHECK-NEXT:       hlfir.assign %[[VAL]] to %[[PRIV]] : f32, !fir.ref<f32>
