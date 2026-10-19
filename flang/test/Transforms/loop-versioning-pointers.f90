! RUN: %flang_fc1 -emit-fir -O3 %s -o - | \
! RUN:   fir-opt --verify-each --flang-licm --loop-versioning | \
! RUN:   FileCheck %s --enable-var-scope
! RUN: %flang -S -O3 -mmlir --mlir-disable-threading \
! RUN:   -mmlir --mlir-print-ir-after=loop-versioning \
! RUN:   %s -o /dev/null 2>&1 | FileCheck %s --enable-var-scope
! Verify that loops over POINTER arrays that are not dummy arguments (local
! variables, derived type components and module variables) are versioned for
! unit stride once LICM has hoisted their descriptor loads out of the loop.
! Descriptors that may change inside the loop must not be versioned.

module loop_versioning_pointers
  implicit none
  type :: holder
    real(8), pointer :: v(:) => null()
  end type
  real(8), pointer :: gp(:) => null()
  interface
    subroutine ext()
    end subroutine
  end interface
contains
  subroutine local_ptr(a, x, n)
    real(8), target, intent(inout) :: a(:,:)
    real(8), target, intent(in) :: x(:,:)
    integer, intent(in) :: n
    real(8), pointer :: p(:), q(:)
    integer :: i, j
    do j = 1, size(a, 2)
      p => a(:, j)
      q => x(:, j)
      do i = 1, n
        p(i) = p(i) + 2.0d0 * q(i)
      end do
    end do
  end subroutine
! CHECK-LABEL: func.func @_QMloop_versioning_pointersPlocal_ptr(
! CHECK: arith.cmpi eq, %{{.*}}#2, %c8{{.*}} : index
! CHECK: arith.cmpi eq, %{{.*}}#2, %c8{{.*}} : index
! CHECK: fir.if
! CHECK: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xf64>>, index) -> !fir.ref<f64>
! CHECK: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xf64>>, index) -> !fir.ref<f64>
! CHECK: } else {
! CHECK: fir.array_coor {{.*}} : (!fir.box<!fir.ptr<!fir.array<?xf64>>>, !fir.shift<1>, i64) -> !fir.ref<f64>

  subroutine component_ptr(h, g, n)
    type(holder), intent(inout) :: h
    type(holder), intent(in) :: g
    integer, intent(in) :: n
    integer :: i
    do i = 1, n
      h%v(i) = h%v(i) + 2.0d0 * g%v(i)
    end do
  end subroutine
! CHECK-LABEL: func.func @_QMloop_versioning_pointersPcomponent_ptr(
! CHECK: arith.cmpi eq, %{{.*}}#2, %c8{{.*}} : index
! CHECK: arith.cmpi eq, %{{.*}}#2, %c8{{.*}} : index
! CHECK: fir.if
! CHECK: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xf64>>, index) -> !fir.ref<f64>
! CHECK: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xf64>>, index) -> !fir.ref<f64>
! CHECK: } else {
! CHECK: fir.array_coor {{.*}} : (!fir.box<!fir.ptr<!fir.array<?xf64>>>, !fir.shift<1>, i64) -> !fir.ref<f64>

  subroutine module_ptr(n, s)
    integer, intent(in) :: n
    real(8), intent(out) :: s
    integer :: i
    s = 0.0d0
    do i = 1, n
      s = s + gp(i)
    end do
  end subroutine
! CHECK-LABEL: func.func @_QMloop_versioning_pointersPmodule_ptr(
! CHECK: arith.cmpi eq, %{{.*}}#2, %c8{{.*}} : index
! CHECK: fir.if
! CHECK: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xf64>>, index) -> !fir.ref<f64>
! CHECK: } else {
! CHECK: fir.array_coor {{.*}} : (!fir.box<!fir.ptr<!fir.array<?xf64>>>, !fir.shift<1>, i64) -> !fir.ref<f64>

  subroutine local_cmplx_ptr(res, k, f, n)
    complex(8), target, intent(inout) :: res(:,:)
    real(8), target, intent(in) :: k(:,:)
    real(8), intent(in) :: f
    integer, intent(in) :: n
    complex(8), pointer :: r(:)
    real(8), pointer :: kx(:)
    integer :: i
    r => res(:, 1)
    kx => k(:, 1)
    do i = 1, n
      r(i) = r(i) + cmplx(f, kx(i), kind=8)
    end do
  end subroutine
! CHECK-LABEL: func.func @_QMloop_versioning_pointersPlocal_cmplx_ptr(
! CHECK-DAG: arith.cmpi eq, %{{.*}}#2, %c16{{.*}} : index
! CHECK-DAG: arith.cmpi eq, %{{.*}}#2, %c8{{.*}} : index
! CHECK: fir.if
! CHECK: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xcomplex<f64>>>, index) -> !fir.ref<complex<f64>>
! CHECK: } else {
! CHECK: fir.array_coor {{.*}} : (!fir.box<!fir.ptr<!fir.array<?xcomplex<f64>>>>, !fir.shift<1>, i64) -> !fir.ref<complex<f64>>

  ! The pointer is re-associated inside the loop.
  subroutine reassociated_in_loop(a, n, s)
    real(8), target, intent(in) :: a(:,:)
    integer, intent(in) :: n
    real(8), intent(out) :: s
    real(8), pointer :: p(:)
    integer :: i
    s = 0.0d0
    p => a(:, 1)
    do i = 1, n
      s = s + p(i)
      p => a(:, 2)
    end do
  end subroutine
! CHECK-LABEL: func.func @_QMloop_versioning_pointersPreassociated_in_loop(
! CHECK-NOT: fir.if
! CHECK: return

  ! The call may re-associate the module pointer.
  subroutine module_ptr_with_call(n, s)
    integer, intent(in) :: n
    real(8), intent(out) :: s
    integer :: i
    s = 0.0d0
    do i = 1, n
      s = s + gp(i)
      call ext()
    end do
  end subroutine
! CHECK-LABEL: func.func @_QMloop_versioning_pointersPmodule_ptr_with_call(
! CHECK-NOT: fir.if
! CHECK: return
end module
