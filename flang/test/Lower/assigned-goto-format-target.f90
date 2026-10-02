! RUN: bbc -emit-fir -o - %s | FileCheck %s

! A FORMAT statement is not a branch target.  Branching to a label that was
! ASSIGN'd from one is not conforming, and the program is meant to reach the
! run-time error rather than jump into the FORMAT statement.

! The only label assigned to j is a FORMAT, so no target survives and no
! branch is generated at all.
! CHECK-LABEL: func.func @_QPfmt_only(
! CHECK:         %[[J:.*]] = fir.declare %arg0
! CHECK:         fir.store %c1{{.*}} to %[[J]]
! CHECK-NOT:     fir.select
! CHECK-NOT:     ^bb
! CHECK:         fir.call @_FortranAReportFatalUserError
! CHECK-NEXT:    fir.unreachable
subroutine fmt_only(j)
  integer :: j
  assign 1 to j
  go to j
1 format("fmt")
end subroutine

! Both labels are assigned to j, but only 20 is a branch target.  The select
! carries exactly one case, for 20; label 1 does not appear.
! CHECK-LABEL: func.func @_QPfmt_and_real(
! CHECK:         %[[J:.*]] = fir.declare %arg0
! CHECK:         fir.store %c1{{.*}} to %[[J]]
! CHECK:         fir.store %c20{{.*}} to %[[J]]
! CHECK:         %[[V:.*]] = fir.load %[[J]]
! CHECK:         fir.select %[[V]] : i32 [20, ^bb[[TGT:[0-9]+]], unit, ^bb[[ERR:[0-9]+]]]
!
! The default destination reports the error and terminates.
! CHECK:       ^bb[[ERR]]:
! CHECK:         fir.call @_FortranAReportFatalUserError
! CHECK-NEXT:    fir.unreachable
!
! The one real target is the PRINT at label 20, which returns normally.
! CHECK:       ^bb[[TGT]]:
! CHECK:         fir.call @_FortranAioBeginExternalListOutput
! CHECK:         return
subroutine fmt_and_real(j)
  integer :: j
  assign 1 to j
  assign 20 to j
  go to j
1 format("fmt")
20 print *, "twenty"
end subroutine

! FORMAT is the only labelled statement that can be assigned and then reach the
! GO TO without being a branch target; every other kind is rejected by semantic
! analysis at the ASSIGN.  The labels below are branch targets of four different
! kinds -- an action statement, the statement that begins an IF construct, the
! statement that begins a DO construct, and the END statement of the subroutine
! -- so every one of them survives and appears as a case of the select.  Each is
! branched to from the same inclusive scope, and the two construct labels name
! the statement that begins the construct rather than the one that ends it, so
! control enters the construct normally instead of jumping into its interior.
! CHECK-LABEL: func.func @_QPbranch_target_kinds(
! CHECK:         fir.select %{{.*}} : i32 [
! CHECK-SAME:      10, ^bb{{[0-9]+}}, 20, ^bb{{[0-9]+}},
! CHECK-SAME:      30, ^bb{{[0-9]+}}, 40, ^bb{{[0-9]+}},
! CHECK-SAME:      unit, ^bb{{[0-9]+}}]
subroutine branch_target_kinds(n)
  integer :: n, j
  assign 10 to j
  assign 20 to j
  assign 30 to j
  assign 40 to j
  go to j
10 continue
20 if (n == 1) then
     print *, "a"
   end if
30 do while (n > 0)
     n = n - 1
   end do
40 end subroutine
