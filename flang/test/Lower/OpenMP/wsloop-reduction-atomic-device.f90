! Device atomic reductions are deferred and handled separately for the device
! runtime. When compiling for the target device, the declare_reduction `atomic`
! region must be left empty so the OpenMP device runtime uses its existing
! buffered cross-team path instead of a host-style atomicrmw fold.
! (Contrast wsloop-reduction-atomic.f90, where the host gets an atomic region.)

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s

subroutine dev_red(a, n, res)
  integer :: n, i
  real :: a(n), res
  res = 0.0
  !$omp target teams distribute parallel do reduction(+:res) map(to:a) map(tofrom:res)
  do i = 1, n
     res = res + a(i)
  end do
end subroutine

! The reduction declaration is still emitted, but with no atomic region on the
! device: only init and combiner, and no llvm.atomicrmw anywhere.
! CHECK-LABEL: omp.declare_reduction @add_reduction_f32 : f32
! CHECK:         combiner
! CHECK-NOT:     atomic {
! CHECK-NOT:     llvm.atomicrmw
