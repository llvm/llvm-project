! Verify that -fdebug-dump-all dumps both symbols and the parse tree, even when semantic errors are present

!----------
! RUN lines
!----------
! RUN: not %flang_fc1 -fdebug-dump-all %s 2>&1 | FileCheck %s

!----------------
! EXPECTED OUTPUT
!----------------
! CHECK: error: Semantic errors in
! CHECK: Flang: parse tree dump
! CHECK: Flang: symbols dump

!-------
! INPUT
!-------
program bad
  real,pointer :: x
  x = null()      ! Error - must be pointer assignment
end
