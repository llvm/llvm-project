! RUN: bbc -emit-fir -o - %s | FileCheck %s

! A DO statement is a valid branch target (F2023 11.2.1), and branching to it
! restarts the loop: the DO construct becomes active again, its bounds are
! re-evaluated and the iteration count re-established (F2023 11.1.7.3 p1,
! 11.1.7.4.1 p1).
!
! A loop whose body branching is self-contained keeps its structured form, with
! the body folded into an scf.execute_region. The loop control statements are
! not part of that body: they may be branched to from outside the loop, so
! their blocks have to stay in the enclosing region. Emitting the DO statement
! inside the wrap gave it a block the enclosing code could not reference.

! The ASSIGN makes a body statement a new block, so the body is wrapped; the
! GOTO targets the loop header from outside the loop.
subroutine branch_to_header(a, b)
  real :: a(10), b(10)
  integer :: m
  go to 80
40 do 42 i = 1, 10
     assign 41 to m
41   a(i) = b(i)
42   continue
80 continue
86 go to 40
end subroutine

! CHECK-LABEL: func.func @_QPbranch_to_header
! The loop header is a branch target, so it starts a block in the function's
! own region -- the same region the branch is emitted from.
! CHECK:         cf.br ^bb[[HEADER:[0-9]+]]
! CHECK:       ^bb[[HEADER]]:
! CHECK:         fir.do_loop
! The body, and only the body, lives in the region.
! CHECK:           scf.execute_region no_inline {
! CHECK:             scf.yield
! CHECK:           }
! CHECK:         }

! Same shape with the ASSIGN target outside the loop: the body needs no blocks
! of its own, so no wrap is emitted at all and the loop stays plain.
subroutine no_wrap_needed(a, b)
  real :: a(10), b(10)
  integer :: m
  go to 80
40 do 42 i = 1, 10
     assign 80 to m
42   a(i) = b(i)
80 continue
86 go to 40
end subroutine

! CHECK-LABEL: func.func @_QPno_wrap_needed
! CHECK-NOT:     scf.execute_region
! CHECK:         fir.do_loop
