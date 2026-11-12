! Verify that -fdebug-dump-all dumps both symbols and the parse tree, even when semantic errors are present

!----------
! RUN lines
!----------
! RUN: not %flang_fc1 -fdebug-dump-all %s 2>&1 | FileCheck %s

! CHECK: error: Semantic errors in
! CHECK: Flang: parse tree dump
! CHECK: Flang: symbols dump

program bad
  type dt(k)
    integer(kind=16) :: k
    integer(kind=16) :: comp
  end type dt
end
