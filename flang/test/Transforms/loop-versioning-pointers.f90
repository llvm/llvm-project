! RUN: %flang_fc1 -emit-fir -O3 %s -o - | \
! RUN:   fir-opt --verify-each --loop-versioning | \
! RUN:   FileCheck %s --enable-var-scope
! RUN: %flang -S -O3 -mmlir --mlir-disable-threading \
! RUN:   -mmlir --mlir-print-ir-after=loop-versioning \
! RUN:   %s -o /dev/null 2>&1 | FileCheck %s --enable-var-scope
! Verify that a loop accessing a local POINTER that is associated with an
! assumed-shape dummy, or with a section of one that keeps the first
! dimension contiguous, is versioned like a loop accessing the dummy itself.
! A loop that is versioned anyway also checks the stride of a local POINTER
! whose pointer assignment is not visible, unless the loop may modify it.
! Other POINTERs are not versioned. Where no loop is versioned, the dummy `a`
! is only there so that the pass has an assumed-shape argument to consider.

module loop_versioning_pointers
  implicit none
  type :: holder
    real(8), pointer :: v(:) => null()
  end type
  real(8), pointer :: gp(:) => null(), hp(:) => null()
  interface
    subroutine ext(p)
      real(8), pointer :: p(:)
    end subroutine
  end interface
contains
  subroutine whole(a, b, n)
    real(8), target :: a(:), b(:)
    integer :: n, i
    real(8), pointer :: p(:), q(:)
    p => a
    q => b
    do i = 1, n
      p(i) = p(i) + q(i)
    end do
  end subroutine
! CHECK-LABEL: func.func @_QMloop_versioning_pointersPwhole(
! CHECK: arith.cmpi eq, %{{.*}}#2, %c8{{.*}} : index
! CHECK: arith.cmpi eq, %{{.*}}#2, %c8{{.*}} : index
! CHECK: fir.if
! CHECK: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xf64>>, index) -> !fir.ref<f64>
! CHECK: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xf64>>, index) -> !fir.ref<f64>
! CHECK: } else {
! CHECK: fir.array_coor {{.*}} : (!fir.box<!fir.ptr<!fir.array<?xf64>>>, !fir.shift<1>, i64) -> !fir.ref<f64>

  subroutine columns(a, x, n)
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
! CHECK-LABEL: func.func @_QMloop_versioning_pointersPcolumns(
! CHECK: arith.cmpi eq, %{{.*}}#2, %c8{{.*}} : index
! CHECK: arith.cmpi eq, %{{.*}}#2, %c8{{.*}} : index
! CHECK: fir.if
! CHECK: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xf64>>, index) -> !fir.ref<f64>
! CHECK: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xf64>>, index) -> !fir.ref<f64>
! CHECK: } else {
! CHECK: fir.array_coor {{.*}} : (!fir.box<!fir.ptr<!fir.array<?xf64>>>, !fir.shift<1>, i64) -> !fir.ref<f64>

  subroutine cmplx_column(res, k, f, n)
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
! CHECK-LABEL: func.func @_QMloop_versioning_pointersPcmplx_column(
! CHECK-DAG: arith.cmpi eq, %{{.*}}#2, %c16{{.*}} : index
! CHECK-DAG: arith.cmpi eq, %{{.*}}#2, %c8{{.*}} : index
! CHECK: fir.if
! CHECK: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xcomplex<f64>>>, index) -> !fir.ref<complex<f64>>
! CHECK: } else {
! CHECK: fir.array_coor {{.*}} : (!fir.box<!fir.ptr<!fir.array<?xcomplex<f64>>>>, !fir.shift<1>, i64) -> !fir.ref<complex<f64>>

  ! A row of a dummy is not contiguous.
  subroutine row(a, b, n)
    real(8), target :: a(:,:), b(:,:)
    integer :: n, i
    real(8), pointer :: p(:), q(:)
    p => a(1, :)
    q => b(1, :)
    do i = 1, n
      p(i) = p(i) + q(i)
    end do
  end subroutine
! CHECK-LABEL: func.func @_QMloop_versioning_pointersProw(
! CHECK-NOT: fir.if
! CHECK: return

  ! The POINTERs are associated with local arrays, not with a dummy.
  subroutine local_target(a, n)
    real(8), intent(inout) :: a(:)
    integer :: n, i
    real(8), target :: t(1000), u(1000)
    real(8), pointer :: p(:), q(:)
    p => t
    q => u
    do i = 1, n
      p(i) = p(i) + q(i)
    end do
    a(1) = p(1)
  end subroutine
! CHECK-LABEL: func.func @_QMloop_versioning_pointersPlocal_target(
! CHECK-NOT: fir.if
! CHECK: return

  subroutine component(a, h, g, n)
    real(8), intent(inout) :: a(:)
    type(holder), intent(inout) :: h
    type(holder), intent(in) :: g
    integer, intent(in) :: n
    integer :: i
    do i = 1, n
      h%v(i) = h%v(i) + g%v(i)
    end do
    a(1) = h%v(1)
  end subroutine
! CHECK-LABEL: func.func @_QMloop_versioning_pointersPcomponent(
! CHECK-NOT: fir.if
! CHECK: return

  subroutine module_ptr(a, n)
    real(8), intent(inout) :: a(:)
    integer, intent(in) :: n
    integer :: i
    do i = 1, n
      gp(i) = gp(i) + hp(i)
    end do
    a(1) = gp(1)
  end subroutine
! CHECK-LABEL: func.func @_QMloop_versioning_pointersPmodule_ptr(
! CHECK-NOT: fir.if
! CHECK: return

  ! The POINTER is re-associated inside the loop.
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

  ! The call may re-associate the POINTERs after their assignment.
  subroutine call_between(a, b, n)
    real(8), target :: a(:), b(:)
    integer :: n, i
    real(8), pointer :: p(:), q(:)
    p => a
    q => b
    call ext(p)
    do i = 1, n
      p(i) = p(i) + q(i)
    end do
  end subroutine
! CHECK-LABEL: func.func @_QMloop_versioning_pointersPcall_between(
! CHECK-NOT: fir.if
! CHECK: return

  ! The loop is versioned for the dummy anyway, so it also checks the stride
  ! of the POINTER that the call associated.
  subroutine ride_along(a, n)
    real(8), intent(inout) :: a(:)
    integer, intent(in) :: n
    real(8), pointer :: q(:)
    integer :: i
    call ext(q)
    do i = 1, n
      a(i) = a(i) + q(i)
    end do
  end subroutine
! CHECK-LABEL: func.func @_QMloop_versioning_pointersPride_along(
! CHECK: arith.cmpi eq, %{{.*}}#2, %c8{{.*}} : index
! CHECK: arith.cmpi eq, %{{.*}}#2, %c8{{.*}} : index
! CHECK: fir.if
! CHECK-NOT: fir.array_coor
! CHECK: } else {
! CHECK: fir.array_coor {{.*}} : (!fir.box<!fir.ptr<!fir.array<?xf64>>>, !fir.shift<1>, i64) -> !fir.ref<f64>

  ! The same, with the loop versioned for a POINTER to a column of a dummy.
  subroutine view_ride_along(th, n)
    real(8), target, intent(in) :: th(:,:)
    integer, intent(in) :: n
    real(8), pointer :: t(:), q(:)
    integer :: i, j
    call ext(q)
    do j = 1, size(th, 2)
      t => th(:, j)
      do i = 1, n
        q(i) = q(i) * t(i)
      end do
    end do
  end subroutine
! CHECK-LABEL: func.func @_QMloop_versioning_pointersPview_ride_along(
! CHECK: arith.cmpi eq, %{{.*}}#2, %c8{{.*}} : index
! CHECK: arith.cmpi eq, %{{.*}}#2, %c8{{.*}} : index
! CHECK: fir.if
! CHECK-NOT: fir.array_coor
! CHECK: } else {
! CHECK: fir.array_coor {{.*}} : (!fir.box<!fir.ptr<!fir.array<?xf64>>>, !fir.shift<1>, i64) -> !fir.ref<f64>

  ! The loop may re-associate the POINTER, so only the dummy is checked.
  subroutine ride_along_modified(a, n)
    real(8), intent(inout) :: a(:)
    integer, intent(in) :: n
    real(8), pointer :: q(:)
    integer :: i
    call ext(q)
    do i = 1, n
      a(i) = a(i) + q(i)
      call ext(q)
    end do
  end subroutine
! CHECK-LABEL: func.func @_QMloop_versioning_pointersPride_along_modified(
! CHECK: arith.cmpi eq, %{{.*}}#2, %c8{{.*}} : index
! CHECK-NOT: arith.cmpi eq
! CHECK: fir.if
! CHECK: fir.array_coor {{.*}} : (!fir.box<!fir.ptr<!fir.array<?xf64>>>, !fir.shift<1>, i64) -> !fir.ref<f64>
! CHECK: } else {

  ! The POINTER is associated with a row, which is not contiguous, so only
  ! the dummy is checked.
  subroutine ride_along_row(a, b, n)
    real(8), intent(inout) :: a(:)
    real(8), target, intent(in) :: b(:,:)
    integer, intent(in) :: n
    real(8), pointer :: q(:)
    integer :: i
    q => b(1, :)
    do i = 1, n
      a(i) = a(i) + q(i)
    end do
  end subroutine
! CHECK-LABEL: func.func @_QMloop_versioning_pointersPride_along_row(
! CHECK: arith.cmpi eq, %{{.*}}#2, %c8{{.*}} : index
! CHECK-NOT: arith.cmpi eq
! CHECK: fir.if
! CHECK: fir.array_coor {{.*}} : (!fir.box<!fir.ptr<!fir.array<?xf64>>>, !fir.shift<1>, i64) -> !fir.ref<f64>
! CHECK: } else {
end module
