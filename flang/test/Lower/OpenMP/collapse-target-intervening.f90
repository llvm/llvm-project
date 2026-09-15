! Lowering of a collapsed imperfect loop nest whose bounds are host-evaluated
! for an enclosing omp.target region. The intervening code's guard and
! terminal-IV restoration consume the omp.target host_eval block arguments
! directly; the omp.target verifier permits this because the arithmetic is
! memory-effect free. This test verifies the nest lowers cleanly and that the
! guard is emitted.

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! CHECK-NOT: not yet implemented

! CHECK-LABEL: func.func @_QPrepro
! CHECK: omp.target
! host_eval confirms the collapse bounds come from the host rather than being
! computed inside the region.
! CHECK-SAME: host_eval(
! CHECK: omp.teams
! CHECK: omp.parallel
! CHECK: omp.distribute
! CHECK: omp.wsloop
! CHECK: omp.loop_nest ({{.*}}) : i32
! The intervening statement "x = x + j" is guarded by an equality compare on
! the inner induction variable, executed inside a fir.if.
! CHECK: arith.cmpi eq
! CHECK: fir.if
! CHECK: hlfir.assign

subroutine repro(n, m, x)
  implicit none
  integer, intent(in) :: n, m
  integer, intent(inout) :: x
  integer :: i, j

  !$omp target teams distribute parallel do collapse(2) map(tofrom:x)
  do i = 1, n
    do j = 1, m
      x = x + 1
    end do
    x = x + j
  end do
end subroutine

! Non-unit inner step: computeLastIV cannot short-circuit to the upper bound and
! must emit the full lb + ((ub-lb)/step)*step arithmetic on the host_eval block
! arguments. The arith.divsi/arith.muli are the distinctive markers of that path
! (absent for unit steps).

! CHECK-LABEL: func.func @_QPrepro_step
! CHECK: omp.target
! CHECK-SAME: host_eval(
! CHECK: omp.loop_nest (%[[SI:.*]], %[[SJ:.*]]) : i32
! CHECK: arith.divsi
! CHECK: arith.muli
! CHECK: arith.cmpi eq, %[[SJ]], %{{.*}} : i32
! CHECK: fir.if
! CHECK: hlfir.assign
! CHECK-NOT: not yet implemented

subroutine repro_step(n, m, x)
  implicit none
  integer, intent(in) :: n, m
  integer, intent(inout) :: x
  integer :: i, j

  !$omp target teams distribute parallel do collapse(2) map(tofrom:x)
  do i = 1, n
    do j = 1, m, 2
      x = x + 1
    end do
    x = x + j
  end do
end subroutine

! collapse(3): the level-0 "after" guard ANDs the equality compares for both
! inner induction variables.

! CHECK-LABEL: func.func @_QPrepro_collapse3
! CHECK: omp.target
! CHECK-SAME: host_eval(
! CHECK: omp.loop_nest (%[[TI:.*]], %[[TJ:.*]], %[[TK:.*]]) : i32
! CHECK: arith.cmpi eq
! CHECK: arith.cmpi eq
! CHECK: arith.andi
! CHECK: fir.if
! CHECK: hlfir.assign
! CHECK-NOT: not yet implemented

subroutine repro_collapse3(n, m, p, x)
  implicit none
  integer, intent(in) :: n, m, p
  integer, intent(inout) :: x
  integer :: i, j, k

  !$omp target teams distribute parallel do collapse(3) map(tofrom:x)
  do i = 1, n
    do j = 1, m
      do k = 1, p
        x = x + 1
      end do
    end do
    x = x + j
  end do
end subroutine

! "before" intervening code is guarded on the inner IV == its lower bound, so it
! runs once per outer iteration. This exercises the guard-on-lower-bound side
! (computeLastIV is not used here).

! CHECK-LABEL: func.func @_QPrepro_before
! CHECK: omp.target
! CHECK-SAME: host_eval(
! CHECK: omp.loop_nest (%[[BI:.*]], %[[BJ:.*]]) : i32
! CHECK: arith.cmpi eq, %[[BJ]], %{{.*}} : i32
! CHECK: fir.if
! CHECK: hlfir.assign
! CHECK-NOT: not yet implemented

subroutine repro_before(n, m, x)
  implicit none
  integer, intent(in) :: n, m
  integer, intent(inout) :: x
  integer :: i, j

  !$omp target teams distribute parallel do collapse(2) map(tofrom:x)
  do i = 1, n
    x = x + 1
    do j = 1, m
      x = x + 1
    end do
  end do
end subroutine
