! Test that -fno-openacc-combined-loop-firstprivate keeps firstprivate on the
! compute construct only, with no implicit loop firstprivate.

! RUN: %flang_fc1 -fopenacc -fno-openacc-combined-loop-firstprivate -emit-hlfir %s -o - | FileCheck %s

subroutine flag_off_scalar
  integer :: i, n, v
  real :: a(10)
  n = 10
  v = 7
  !$acc parallel loop firstprivate(v)
  do i = 1, n
    a(i) = v
  end do
end subroutine

! CHECK-LABEL: func.func @_QPflag_off_scalar
! CHECK: %[[FP_V:.*]] = acc.firstprivate varPtr(%{{.*}} : !fir.ref<i32>) recipe({{.*}}) name("v") -> !fir.ref<i32>
! CHECK: acc.parallel combined(loop) {{.*}}firstprivate(%[[FP_V]] : !fir.ref<i32>)
! CHECK-NOT: acc.firstprivate {{.*}} implicit(true)
! CHECK: acc.loop combined(parallel)
! CHECK: } inclusiveUpperbound(array<i1: true>) independent

subroutine flag_off_array
  integer :: i, n
  real :: b(10)
  n = 10
  !$acc parallel loop firstprivate(b)
  do i = 1, n
    b(i) = 1.0
  end do
end subroutine

! CHECK-LABEL: func.func @_QPflag_off_array
! CHECK: acc.firstprivate varPtr(%{{.*}} : !fir.ref<!fir.array<10xf32>>) recipe({{.*}}) name("b")
! CHECK: acc.parallel combined(loop) {{.*}}firstprivate
! CHECK-NOT: acc.firstprivate {{.*}} implicit(true)
! CHECK: acc.loop combined(parallel)

subroutine flag_off_serial_loop
  integer :: i, n, v
  real :: a(10)
  n = 10
  v = 7
  !$acc serial loop firstprivate(v)
  do i = 1, n
    a(i) = v
  end do
end subroutine

! CHECK-LABEL: func.func @_QPflag_off_serial_loop
! CHECK: acc.firstprivate varPtr(%{{.*}} : !fir.ref<i32>) recipe({{.*}}) name("v")
! CHECK: acc.serial combined(loop) {{.*}}firstprivate
! CHECK-NOT: acc.firstprivate {{.*}} implicit(true)
! CHECK: acc.loop combined(serial)
