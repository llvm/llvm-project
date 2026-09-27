! Regression test: a collapsed imperfect loop nest with genuine intervening
! code, whose loop bounds are host-evaluated for an enclosing omp.target SPMD
! region, is not yet supported and must be diagnosed cleanly rather than
! producing IR that fails the omp.target verifier.
!
! The intervening statement "x = x + j" runs at intermediate nest level. Its
! guard and terminal-IV restoration would perform arith.cmpi/arith.addi on the
! collapsed loop bounds, which are omp.target host_eval block arguments in this
! context -- an illegal use. Until that path is implemented, emit a "not yet
! implemented" message.

! RUN: not %flang_fc1 -emit-hlfir -fopenmp %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: collapsed loop nest with intervening code whose loop bounds are evaluated on the host for an enclosing 'target' region
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
