! Test that mixing a volatile and a non-volatile scalar by-value intrinsic
! reduction of the same operator/element type produces two distinct
! omp.declare_reduction ops, in either source order. Only the non-volatile f32
! one has an atomic region; f64 is too wide for one.

! RUN: bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

! Reduction ops are emitted in reverse source order; the CHECKs follow that.

! CHECK-LABEL: omp.declare_reduction @add_reduction_volatile_f64 : f64
! CHECK-NOT:     atomic {
! CHECK-NOT:     llvm.atomicrmw
! CHECK-LABEL: omp.declare_reduction @add_reduction_f64 : f64
! CHECK-NOT:     llvm.atomicrmw
! CHECK-LABEL: omp.declare_reduction @add_reduction_f32 : f32
! CHECK:         llvm.atomicrmw fadd %{{.*}}, %{{.*}} monotonic : !llvm.ptr, f32
! CHECK-LABEL: omp.declare_reduction @add_reduction_volatile_f32 : f32
! CHECK-NOT:     atomic {
! CHECK-NOT:     llvm.atomicrmw

! Volatile variable reduced first, then the non-volatile one.
! CHECK-LABEL: func.func @_QPred_vol_then_nonvol
! CHECK:         omp.wsloop {{.*}}reduction(@add_reduction_volatile_f32 %{{.*}} -> %{{.*}} : !fir.ref<f32, volatile>)
! CHECK:         omp.wsloop {{.*}}reduction(@add_reduction_f32 %{{.*}} -> %{{.*}} : !fir.ref<f32>)

subroutine red_vol_then_nonvol(a, n, vres, res)
  integer :: n, i
  real, volatile :: vres
  real :: res, a(n)
  vres = 0.0
  res = 0.0
  !$omp parallel do reduction(+:vres)
  do i = 1, n
    vres = vres + a(i)
  end do
  !$omp parallel do reduction(+:res)
  do i = 1, n
    res = res + a(i)
  end do
end subroutine

! Non-volatile variable reduced first, then the volatile one.
! CHECK-LABEL: func.func @_QPred_nonvol_then_vol
! CHECK:         omp.wsloop {{.*}}reduction(@add_reduction_f64 %{{.*}} -> %{{.*}} : !fir.ref<f64>)
! CHECK:         omp.wsloop {{.*}}reduction(@add_reduction_volatile_f64 %{{.*}} -> %{{.*}} : !fir.ref<f64, volatile>)

subroutine red_nonvol_then_vol(a, n, res, vres)
  integer :: n, i
  real(8), volatile :: vres
  real(8) :: res, a(n)
  vres = 0.0
  res = 0.0
  !$omp parallel do reduction(+:res)
  do i = 1, n
    res = res + a(i)
  end do
  !$omp parallel do reduction(+:vres)
  do i = 1, n
    vres = vres + a(i)
  end do
end subroutine
