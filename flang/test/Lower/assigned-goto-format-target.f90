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
