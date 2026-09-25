! An IF body ending in a CYCLE must still lower to an acc.loop with a control
! clause: without the induction variable and bounds there is nothing to map
! onto gangs and threads, so the loop could only be emitted as a serial kernel.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

subroutine trailing_cycle(n, rho, v)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: rho(n)
  real(8), intent(out) :: v(n)
  integer :: i

  !$acc parallel loop present(rho, v)
  do i = 1, n
     if (rho(i) <= 1.0d-10) then
        v(i) = 0.0d0
        cycle
     end if
     v(i) = 1.0d0 / rho(i)
  end do
end subroutine trailing_cycle

! CHECK-LABEL: func.func @_QPtrailing_cycle
! CHECK: acc.parallel
! CHECK: acc.loop
! CHECK-SAME: control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32)
! CHECK: fir.if
! CHECK-NOT: cf.cond_br
