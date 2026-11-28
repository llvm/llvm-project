! Test that mixing a volatile and a non-volatile scalar by-value intrinsic
! reduction of the same operator/element type produces two distinct
! omp.declare_reduction ops, in either source order. Only the non-volatile ones
! have an atomic region; the f64 operands are dummies, so theirs is the
! `_unaligned` declaration without one.

! RUN: bbc --strict-fir-volatile-verifier -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

! Reduction ops are emitted in reverse source order; the CHECKs follow that.

! CHECK-LABEL: omp.declare_reduction @add_reduction_volatile_f64 : f64
! CHECK-NOT:     atomic {
! CHECK-NOT:     llvm.atomicrmw
! CHECK-LABEL: omp.declare_reduction @add_reduction_unaligned_f64 : f64
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
! CHECK:         omp.wsloop {{.*}}reduction(@add_reduction_unaligned_f64 %{{.*}} -> %{{.*}} : !fir.ref<f64>)
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

! A volatile array is boxed and reduced by reference; the box reference keeps
! the volatile qualifier.
! CHECK-LABEL: func.func @_QPred_vol_array
! CHECK:         omp.wsloop {{.*}}reduction(byref @add_reduction_byref_box_Uxf32 %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.array<?xf32>, volatile>, volatile>)

subroutine red_vol_array(n, r)
  integer :: n, i
  real, volatile :: r(n)
  !$omp parallel do reduction(+:r)
  do i = 1, n
    r(i) = r(i) + 1.0
  end do
end subroutine
