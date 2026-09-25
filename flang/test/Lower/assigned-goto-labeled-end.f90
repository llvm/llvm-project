! RUN: bbc -pft-test -o - %s | FileCheck %s

! The END statement of a program unit may carry a label and is a branch target
! (F2023 11.2.1 p1 lists end-subroutine-stmt, end-function-stmt and
! end-program-stmt).  Label analysis visits that statement in advance, before
! the statement visitor reaches it, so its recorded position has to be supplied
! explicitly; otherwise lowering fails to recognise it and the assigned GO TO
! silently loses the edge.  These checks pin the behaviour at the PFT layer,
! where the target is marked -- the generated select alone cannot see it,
! because ASSIGN also marks blocks for the labels it names.

! Two units in one file, each with a labeled END.  The second unit's END must
! not be recorded under the first unit's position.
! CHECK-LABEL: Subroutine two_first:
! CHECK:         AssignedGotoStmt! -> [[E1:[0-9]+]]: go to j
! CHECK:         [[E1]] ^EndSubroutineStmt: 41 end subroutine
subroutine two_first(j)
  integer :: j
  assign 41 to j
  go to j
41 end subroutine

! CHECK-LABEL: Subroutine two_second:
! CHECK:         AssignedGotoStmt! -> [[E2:[0-9]+]]: go to j
! CHECK:         [[E2]] ^EndSubroutineStmt: 42 end subroutine
subroutine two_second(j)
  integer :: j
  assign 42 to j
  go to j
42 end subroutine

! An explicit label list naming a labeled END that is never ASSIGN'd.  Nothing
! marks the target except the GO TO itself, so this shape depends only on the
! recorded classification.
! CHECK-LABEL: Subroutine listed_end:
! CHECK:         AssignedGotoStmt! -> [[E3:[0-9]+]]: go to j,(43)
! CHECK:         [[E3]] ^EndSubroutineStmt: 43 end subroutine
subroutine listed_end(j)
  integer :: j
  go to j, (43)
43 end subroutine

! A labeled END of an internal subprogram, reached through the host's CONTAINS.
! CHECK-LABEL: Subroutine inner:
! CHECK:         AssignedGotoStmt! -> [[E4:[0-9]+]]: go to j
! CHECK:         [[E4]] ^EndSubroutineStmt: 44 end subroutine
program host
  call inner(1)
contains
  subroutine inner(j)
    integer :: j
    assign 44 to j
    go to j
44 end subroutine
end program
