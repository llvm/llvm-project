! A combined `acc parallel loop` whose default parallelism resolves to
! `independent` and whose body branching keeps it unstructured.
!
! The GOTO jumps backwards, so the body reaches the GOTO again from its own
! target and the loop is not proven to terminate. It stays unstructured and the
! directive cannot take it over. The patterns that do keep their branching
! confined live in ../acc-unstructured-combined-construct.f90.
!
! A GOTO leaving the loop would not serve here: it is rejected on both paths,
! as is EXIT or RETURN in a combined construct.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s --check-prefix=CBACK-OK
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir --emit-independent-loops-as-unstructured=false %s -o - 2>&1 | FileCheck %s --check-prefix=CBACK

subroutine test_combined_backward_goto(a, n)
  integer :: n, i
  real :: a(n)

  !$acc parallel loop
  do i = 1, n
20  continue
    a(i) = a(i) * 2.0
    if (a(i) < 100.0) goto 20
  end do
end subroutine

! CBACK: not yet implemented: unstructured do loop in combined acc construct

! By default the loop still lowers, but without its bounds on the op: the
! directive did not take it over. The branching stays raw in the loop's own
! region rather than being folded into one, which is what a loop that is
! unstructured throughout gets.
! CBACK-OK-LABEL: func.func @_QPtest_combined_backward_goto
! CBACK-OK:         acc.parallel combined(loop)
! CBACK-OK:           acc.loop combined(parallel) private({{.*}}) {
! CBACK-OK-NOT:         control(
! CBACK-OK-NOT:         scf.execute_region
! CBACK-OK:             cf.cond_br
! CBACK-OK-NOT:         scf.execute_region
! CBACK-OK:           } independent  unstructured
