! Regression test: a collapsed loop nest written with labeled DO loops must
! lower successfully when the loop bounds are host-evaluated under omp.target.
!
! A labeled DO loop leaves a no-op ContinueStmt for its terminating labeled
! statement (in addition to the EndDoStmt). That CONTINUE must not be treated
! as intervening code by the collapsed-loop-nest lowering: doing so emitted a
! guarded "after" region containing arithmetic (arith.addi) on the loop upper
! bound, which is an illegal use of an omp.target host_eval block argument and
! tripped the MLIR verifier ("host_eval argument illegal use in 'arith.addi'").
!
! Before the fix, each of the three target regions below failed to verify.

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPinital
subroutine inital(n, m)
  implicit none
  integer, intent(in) :: n, m
  integer :: i, j

  ! Exercises the labeled-DO collapsed-nest lowering path; verified via its
  ! collapsed loop_nest below.
  ! CHECK: omp.loop_nest {{.*}} collapse(2)
  !$omp target parallel do collapse(2)
  do 10 j = 1, n
  do 11 i = 1, m
  11 end do
  10 end do

  ! CHECK: omp.target {{.*}}host_eval(
  ! CHECK: omp.loop_nest {{.*}} collapse(2)
  !$omp target teams distribute parallel do collapse(2)
  do 20 j = 1, n
  do 21 i = 1, m
  21 end do
  20 end do

  ! CHECK: omp.target {{.*}}host_eval(
  ! CHECK: omp.loop_nest {{.*}} collapse(2)
  !$omp target teams distribute collapse(2)
  do 30 j = 1, n
  do 31 i = 1, m
  31 end do
  30 end do
end subroutine inital
