! 64-bit by-value reductions get an atomic region only when the operand's
! natural alignment is known at the clause. Otherwise they bind a separate
! `_unaligned` declaration without one: a conforming EQUIVALENCE, or an actual
! argument from one, can leave an 8-byte variable only 4-byte aligned, and a
! misaligned atomicrmw can split a cache line or fault.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s | FileCheck %s

! Declarations are emitted in reverse order of first use.

! Not provably aligned: dummy, EQUIVALENCE (directly or in the host), COMMON,
! ASSOCIATE name, and a host-associated dummy.
! CHECK-LABEL: omp.declare_reduction @add_reduction_unaligned_f64 : f64
! CHECK-NOT:     llvm.atomicrmw

! Aligned: locals, module variables (including BIND(C)), and host-associated
! locals.
! CHECK-LABEL: omp.declare_reduction @max_f64 : f64
! CHECK:         llvm.atomicrmw fmax %{{.*}}, %{{.*}} monotonic : !llvm.ptr, f64
! CHECK-LABEL: omp.declare_reduction @max_i64 : i64
! CHECK:         llvm.atomicrmw max %{{.*}}, %{{.*}} monotonic : !llvm.ptr, i64
! CHECK-LABEL: omp.declare_reduction @add_reduction_i64 : i64
! CHECK:         llvm.atomicrmw add %{{.*}}, %{{.*}} monotonic : !llvm.ptr, i64

! Volatile 64-bit operands are never atomic and need no alignment split.
! CHECK-LABEL: omp.declare_reduction @add_reduction_volatile_f64 : f64
! CHECK-NOT:     llvm.atomicrmw
! CHECK-LABEL: omp.declare_reduction @add_reduction_f64 : f64
! CHECK:         llvm.atomicrmw fadd %{{.*}}, %{{.*}} monotonic : !llvm.ptr, f64

module m
  real(8) :: msum
  real(8), bind(c, name="cvar") :: csum
contains
  subroutine mod_var(a, n)
    integer :: n, i
    real(8) :: a(n)
    !$omp parallel do reduction(+:msum)
    do i = 1, n
      msum = msum + a(i)
    end do
  end subroutine
end module

! CHECK-LABEL: func.func @_QPvol
! CHECK:         reduction(@add_reduction_volatile_f64
subroutine vol(a, n)
  integer :: n, i
  real(8) :: a(n)
  real(8), volatile :: vs
  vs = 0
  !$omp parallel do reduction(+:vs)
  do i = 1, n
    vs = vs + a(i)
  end do
end subroutine

! CHECK-LABEL: func.func @_QPlocal
! CHECK:         reduction(@add_reduction_f64
subroutine local(a, n)
  integer :: n, i
  real(8) :: a(n), s
  s = 0
  !$omp parallel do reduction(+:s)
  do i = 1, n
    s = s + a(i)
  end do
end subroutine

! CHECK-LABEL: func.func @_QPsaved
! CHECK:         reduction(@add_reduction_f64
subroutine saved(a, n)
  integer :: n, i
  real(8) :: a(n)
  real(8), save :: s = 0
  !$omp parallel do reduction(+:s)
  do i = 1, n
    s = s + a(i)
  end do
end subroutine

! CHECK-LABEL: func.func @_QPuse_assoc
! CHECK:         reduction(@add_reduction_f64
subroutine use_assoc(a, n)
  use m, only: msum
  integer :: n, i
  real(8) :: a(n)
  !$omp parallel do reduction(+:msum)
  do i = 1, n
    msum = msum + a(i)
  end do
end subroutine

! CHECK-LABEL: func.func @_QPhost_local
! CHECK:         reduction(@add_reduction_f64
subroutine host_local(a, n)
  integer :: n, i
  real(8) :: a(n), s
  s = 0
  call inner()
contains
  subroutine inner()
    !$omp parallel do reduction(+:s)
    do i = 1, n
      s = s + a(i)
    end do
  end subroutine
end subroutine

! CHECK-LABEL: func.func @_QPbindc
! CHECK:         reduction(@add_reduction_f64
subroutine bindc(a, n)
  use m, only: csum
  integer :: n, i
  real(8) :: a(n)
  !$omp parallel do reduction(+:csum)
  do i = 1, n
    csum = csum + a(i)
  end do
end subroutine

! CHECK-LABEL: func.func @_QPint64
! CHECK:         reduction(@add_reduction_i64 {{.*}}, @max_i64
subroutine int64(n)
  integer :: n, i
  integer(8) :: s, m
  s = 0; m = 0
  !$omp parallel do reduction(+:s) reduction(max:m)
  do i = 1, n
    s = s + i
    m = max(m, int(i, 8))
  end do
end subroutine

! CHECK-LABEL: func.func @_QPreal64_max
! CHECK:         reduction(@max_f64
subroutine real64_max(a, n)
  integer :: n, i
  real(8) :: a(n), m
  m = 0
  !$omp parallel do reduction(max:m)
  do i = 1, n
    m = max(m, a(i))
  end do
end subroutine

! CHECK-LABEL: func.func @_QPdummy
! CHECK:         reduction(@add_reduction_unaligned_f64
subroutine dummy(a, n, s)
  integer :: n, i
  real(8) :: a(n), s
  !$omp parallel do reduction(+:s)
  do i = 1, n
    s = s + a(i)
  end do
end subroutine

! CHECK-LABEL: func.func @_QPequiv
! CHECK:         reduction(@add_reduction_unaligned_f64
subroutine equiv(a, n)
  integer :: n, i
  real(8) :: a(n), s
  real :: r(3)
  equivalence (s, r(2))
  s = 0
  !$omp parallel do reduction(+:s)
  do i = 1, n
    s = s + a(i)
  end do
end subroutine

! CHECK-LABEL: func.func @_QPcommon_var
! CHECK:         reduction(@add_reduction_unaligned_f64
subroutine common_var(a, n)
  integer :: n, i, k
  real(8) :: a(n), s
  common /blk/ k, s
  s = 0
  !$omp parallel do reduction(+:s)
  do i = 1, n
    s = s + a(i)
  end do
end subroutine

! CHECK-LABEL: func.func @_QPassoc
! CHECK:         reduction(@add_reduction_unaligned_f64
subroutine assoc(a, n)
  type t
    sequence
    integer :: k
    real(8) :: s
  end type
  integer :: n, i
  real(8) :: a(n)
  type(t) :: v
  v%s = 0
  associate (s => v%s)
    !$omp parallel do reduction(+:s)
    do i = 1, n
      s = s + a(i)
    end do
  end associate
end subroutine

! CHECK-LABEL: func.func @_QPhost_dummy
! CHECK:         reduction(@add_reduction_unaligned_f64
subroutine host_dummy(a, n, s)
  integer :: n, i
  real(8) :: a(n), s
  call inner()
contains
  subroutine inner()
    !$omp parallel do reduction(+:s)
    do i = 1, n
      s = s + a(i)
    end do
  end subroutine
end subroutine

! CHECK-LABEL: func.func @_QPhost_equiv
! CHECK:         reduction(@add_reduction_unaligned_f64
subroutine host_equiv(a, n)
  integer :: n, i
  real(8) :: a(n), s
  real :: r(3)
  equivalence (s, r(2))
  s = 0
  call inner()
contains
  subroutine inner()
    !$omp parallel do reduction(+:s)
    do i = 1, n
      s = s + a(i)
    end do
  end subroutine
end subroutine
