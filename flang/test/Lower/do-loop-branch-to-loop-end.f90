! RUN: bbc -emit-fir -o - %s | FileCheck %s

! A loop whose body branching is self-contained keeps its structured form, with
! the body folded into an scf.execute_region. The EndDoStmt is not part of that
! body: it is emitted as the structured loop's terminator, so no block is
! created for it.
!
! That is fine for a branch from the body, which is CYCLE-like and lands on the
! wrap's boundary. A branch from outside the loop has nowhere to land, so such a
! loop stays unstructured and is lowered as raw blocks.

! The GOTO targets the EndDoStmt from outside the loop.
subroutine branch_to_loop_end(a)
  real :: a(10)
  integer :: i
  do i = 1, 10
     a(i) = 0.0
20 end do
  go to 20
end subroutine

! CHECK-LABEL: func.func @_QPbranch_to_loop_end
! Raw blocks, so the branch has a block to target.
! CHECK:         cf.br ^bb1
! CHECK:       ^bb1:
! CHECK-NOT:     fir.do_loop
! CHECK-NOT:     scf.execute_region

! The same branch from inside the body is CYCLE-like, so the loop keeps its
! structured form. The ASSIGN makes a body statement a new block, so the body is
! wrapped.
subroutine cycle_to_loop_end(a, b)
  real :: a(10), b(10)
  integer :: i, m
  do 42 i = 1, 10
     assign 41 to m
41   a(i) = b(i)
     go to 42
42 end do
end subroutine

! CHECK-LABEL: func.func @_QPcycle_to_loop_end
! CHECK:         fir.do_loop
! CHECK:           scf.execute_region no_inline {
! CHECK:             scf.yield
! CHECK:           }
! CHECK:         }
