! Test that scalar by-value intrinsic reductions emit an `atomic` region in the
! omp.declare_reduction op, enabling the runtime's barrier-free atomic method.
! Reduction ops are emitted in reverse source order; the CHECKs follow that.

! RUN: bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp --force-byref-reduction -o - %s | FileCheck %s --check-prefix=BYREF --implicit-check-not='atomic {' --implicit-check-not=llvm.atomicrmw

! BYREF: omp.declare_reduction @add_reduction_byref_f64
! BYREF: omp.declare_reduction @add_reduction_byref_f32
! BYREF: func.func @_QPred_add_real

! A scalar allocatable is by-ref (descriptor), so it gets no atomic region
! despite having an intrinsic element type.
! CHECK-LABEL: omp.declare_reduction @add_reduction_byref_box_heap_f32
! CHECK-NOT:     llvm.atomicrmw
! A logical '.and.' reduction is a trivial type but neither integer nor float,
! so it gets no atomic region.
! CHECK-LABEL: omp.declare_reduction @and_reduction_l32 : !fir.logical<4>
! CHECK-NOT:     llvm.atomicrmw

! CHECK-LABEL: omp.declare_reduction @max_f64 : f64
! CHECK:         llvm.atomicrmw fmax %{{.*}}, %{{.*}} monotonic : !llvm.ptr, f64
! CHECK-LABEL: omp.declare_reduction @min_f32 : f32
! CHECK:         llvm.atomicrmw fmin %{{.*}}, %{{.*}} monotonic : !llvm.ptr, f32
! CHECK-LABEL: omp.declare_reduction @ieor_i32 : i32
! CHECK:         llvm.atomicrmw _xor %{{.*}}, %{{.*}} monotonic : !llvm.ptr, i32
! CHECK-LABEL: omp.declare_reduction @ior_i32 : i32
! CHECK:         llvm.atomicrmw _or %{{.*}}, %{{.*}} monotonic : !llvm.ptr, i32
! CHECK-LABEL: omp.declare_reduction @min_i32 : i32
! CHECK:         llvm.atomicrmw min %{{.*}}, %{{.*}} monotonic : !llvm.ptr, i32
! CHECK-LABEL: omp.declare_reduction @add_reduction_f64 : f64
! CHECK:         llvm.atomicrmw fadd %{{.*}}, %{{.*}} monotonic : !llvm.ptr, f64
! CHECK-LABEL: omp.declare_reduction @add_reduction_i64 : i64
! CHECK:         llvm.atomicrmw add %{{.*}}, %{{.*}} monotonic : !llvm.ptr, i64
! CHECK-LABEL: omp.declare_reduction @add_reduction_i16 : i16
! CHECK:         llvm.atomicrmw add %{{.*}}, %{{.*}} monotonic : !llvm.ptr, i16
! CHECK-LABEL: omp.declare_reduction @add_reduction_i8 : i8
! CHECK:         llvm.atomicrmw add %{{.*}}, %{{.*}} monotonic : !llvm.ptr, i8

! A complex '+' reduction is a trivial type but neither integer nor float, so no
! atomic region is emitted (the runtime falls back to critical/tree).
! CHECK-LABEL: omp.declare_reduction @add_reduction_z32 : complex<f32>
! CHECK-NOT:     llvm.atomicrmw

! A multiply reduction has no corresponding atomicrmw bin-op, so no atomic region is
! emitted (the runtime falls back to critical/tree).
! CHECK-LABEL: omp.declare_reduction @multiply_reduction_f32 : f32
! CHECK-NOT:     llvm.atomicrmw

! CHECK-LABEL: omp.declare_reduction @iand_i32 : i32
! CHECK:         llvm.atomicrmw _and %{{.*}}, %{{.*}} monotonic : !llvm.ptr, i32

! CHECK-LABEL: omp.declare_reduction @max_i32 : i32
! CHECK:         llvm.atomicrmw max %{{.*}}, %{{.*}} monotonic : !llvm.ptr, i32

! CHECK-LABEL: omp.declare_reduction @add_reduction_i32 : i32
! CHECK:         llvm.atomicrmw add %{{.*}}, %{{.*}} monotonic : !llvm.ptr, i32

! CHECK-LABEL: omp.declare_reduction @add_reduction_f32 : f32
! CHECK:       atomic {
! CHECK:       ^bb0(%[[LHS:.*]]: !fir.ref<f32>, %[[RHS:.*]]: !fir.ref<f32>):
! CHECK:         %[[V:.*]] = fir.load %[[RHS]] : !fir.ref<f32>
! CHECK:         %[[P:.*]] = fir.convert %[[LHS]] : (!fir.ref<f32>) -> !llvm.ptr
! CHECK:         llvm.atomicrmw fadd %[[P]], %[[V]] monotonic : !llvm.ptr, f32
! CHECK:         omp.yield

subroutine red_add_real(a, n, res)
  integer :: n, i
  real :: a(n), res
  res = 0.0
  !$omp parallel do reduction(+:res)
  do i = 1, n
     res = res + a(i)
  end do
end subroutine

subroutine red_add_int(a, n, res)
  integer :: n, i
  integer :: a(n), res
  res = 0
  !$omp parallel do reduction(+:res)
  do i = 1, n
     res = res + a(i)
  end do
end subroutine

subroutine red_max_int(a, n, res)
  integer :: n, i
  integer :: a(n), res
  res = 0
  !$omp parallel do reduction(max:res)
  do i = 1, n
     res = max(res, a(i))
  end do
end subroutine

subroutine red_iand_int(a, n, res)
  integer :: n, i
  integer :: a(n), res
  res = -1
  !$omp parallel do reduction(iand:res)
  do i = 1, n
     res = iand(res, a(i))
  end do
end subroutine

subroutine red_mul_real(a, n, res)
  integer :: n, i
  real :: a(n), res
  res = 1.0
  !$omp parallel do reduction(*:res)
  do i = 1, n
     res = res * a(i)
  end do
end subroutine

subroutine red_add_complex(a, n, res)
  integer :: n, i
  complex :: a(n), res
  res = (0.0, 0.0)
  !$omp parallel do reduction(+:res)
  do i = 1, n
     res = res + a(i)
  end do
end subroutine

! Cover the remaining bin-ops and supported widths without an entity matrix.
subroutine red_ops_and_widths(s8, s16, s64, sreal, lo, bits, parity, rlo, rhi)
  integer(1) :: s8
  integer(2) :: s16
  integer(8) :: s64
  real(8) :: sreal, rhi
  integer :: lo, bits, parity, i
  real :: rlo
  !$omp parallel do reduction(+:s8,s16,s64,sreal) reduction(min:lo) &
  !$omp& reduction(ior:bits) reduction(ieor:parity) reduction(min:rlo) reduction(max:rhi)
  do i = 1, 10
    s8 = s8 + 1_1
    s16 = s16 + 1_2
    s64 = s64 + 1_8
    sreal = sreal + 1.0_8
    lo = min(lo, i)
    bits = ior(bits, i)
    parity = ieor(parity, i)
    rlo = min(rlo, real(i))
    rhi = max(rhi, real(i, 8))
  end do
end subroutine

! A logical '.and.' reduction (trivial but not int/float) must not get atomic.
subroutine red_logical(truth, n)
  logical :: truth
  integer :: n, i
  !$omp parallel do reduction(.and.:truth)
  do i = 1, n
    truth = truth .and. (i > 0)
  end do
end subroutine

! A scalar allocatable is by-ref (descriptor) and must not get atomic.
subroutine red_alloc(x, a, n)
  real, allocatable :: x
  integer :: n, i
  real :: a(n)
  !$omp parallel do reduction(+:x)
  do i = 1, n
    x = x + a(i)
  end do
end subroutine
