! Real DO controls are a Fortran extension. An implicit acc.loop (a DO inside
! a compute construct with no !$acc loop) converts a non-integer control to
! the induction variable's integer type. An integer control is left unchanged.
! An explicit !$acc loop is left unchanged.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

subroutine real_bounds()
  complex :: a(2), b
  integer :: i, n
  a = (1, 3)
  b = (5, 6)
  n = 0
  !$acc serial
  do i = a(2)%re, b%im
    n = n + 1
  end do
  !$acc end serial
end subroutine

! CHECK-LABEL: func.func @_QPreal_bounds
! CHECK: %[[RB_LB:.*]] = fir.convert {{.*}} : (f32) -> i32
! CHECK: %[[RB_UB:.*]] = fir.convert {{.*}} : (f32) -> i32
! CHECK: %[[RB_STEP:.*]] = arith.constant 1 : i32
! CHECK: acc.loop {{.*}} control(%{{.*}} : i32) = (%[[RB_LB]] : i32) to (%[[RB_UB]] : i32) step (%[[RB_STEP]] : i32)

subroutine real_step(r, s, t)
  real :: r, s, t
  integer :: i, n
  n = 0
  !$acc serial
  do i = r, s, t
    n = n + 1
  end do
  !$acc end serial
end subroutine

! CHECK-LABEL: func.func @_QPreal_step
! CHECK: %[[RS_LB:.*]] = fir.convert {{.*}} : (f32) -> i32
! CHECK: %[[RS_UB:.*]] = fir.convert {{.*}} : (f32) -> i32
! CHECK: %[[RS_STEP:.*]] = fir.convert {{.*}} : (f32) -> i32
! CHECK: acc.loop {{.*}} control(%{{.*}} : i32) = (%[[RS_LB]] : i32) to (%[[RS_UB]] : i32) step (%[[RS_STEP]] : i32)
