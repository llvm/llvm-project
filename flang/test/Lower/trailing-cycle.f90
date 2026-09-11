! RUN: %flang_fc1 -fdebug-dump-pft %s 2>&1 | FileCheck %s

! A CYCLE that is the last statement of the body of its own DO is a no-op, and
! is deleted so that the DO is not marked unstructured. A trailing `!' on a
! construct name in the dump marks it unstructured.

! CHECK: 1 Subroutine trailing_cycle
subroutine trailing_cycle(a, n)
  integer :: n, i, j
  real :: a(n)

  ! The CYCLE is deleted; the DoConstruct stays structured.
  ! CHECK:   <<DoConstruct>> -> 5
  ! CHECK:     1 NonLabelDoStmt -> 4: do i = 1, n
  ! CHECK:     2 ^AssignmentStmt: a(i) = 1.0
  ! CHECK:     4 EndDoStmt -> 1: end do
  ! CHECK:   <<End DoConstruct>>
  do i = 1, n
    a(i) = 1.0
    cycle
  end do

  ! Same, with the CYCLE naming its own construct.
  ! CHECK:   <<DoConstruct>> -> 9
  ! CHECK:     5 NonLabelDoStmt -> 8: loop: do i = 1, n
  ! CHECK:     6 ^AssignmentStmt: a(i) = 2.0
  ! CHECK:     8 EndDoStmt -> 5: end do loop
  ! CHECK:   <<End DoConstruct>>
  loop: do i = 1, n
    a(i) = 2.0
    cycle loop
  end do loop

  ! The CYCLE is the whole body.
  ! CHECK:   <<DoConstruct>> -> 12
  ! CHECK:     9 NonLabelDoStmt -> 11: do i = 1, n
  ! CHECK:     11 ^EndDoStmt -> 9: end do
  ! CHECK:   <<End DoConstruct>>
  do i = 1, n
    cycle
  end do

  ! A CYCLE for an outer construct is a real branch and is kept.
  ! CHECK:   <<DoConstruct>> -> 18
  ! CHECK:     12 NonLabelDoStmt -> 17: outer: do i = 1, n
  ! CHECK:     <<DoConstruct!>> -> 17
  ! CHECK:       13 ^NonLabelDoStmt -> 16: do j = 1, n
  ! CHECK:       14 ^AssignmentStmt: a(j) = 3.0
  ! CHECK:       15 CycleStmt! -> 17: cycle outer
  ! CHECK:       16 ^EndDoStmt -> 13: end do
  ! CHECK:     <<End DoConstruct!>>
  ! CHECK:     17 ^EndDoStmt -> 12: end do outer
  ! CHECK:   <<End DoConstruct>>
  outer: do i = 1, n
    do j = 1, n
      a(j) = 3.0
      cycle outer
    end do
  end do outer

  ! A labeled CYCLE may be a branch target and is kept.
  ! CHECK:   <<DoConstruct!>> -> 25
  ! CHECK:     18 NonLabelDoStmt -> 24: do i = 1, n
  ! CHECK:     <<IfConstruct>> -> 23
  ! CHECK:       19 ^IfStmt [negate] -> 23: if(a(i) > 0.0) goto 10
  ! CHECK:       22 ^AssignmentStmt: a(i) = 4.0
  ! CHECK:       21 EndIfStmt
  ! CHECK:     <<End IfConstruct>>
  ! CHECK:     23 CycleStmt! -> 24: 10 cycle
  ! CHECK:     24 ^EndDoStmt -> 18: end do
  ! CHECK:   <<End DoConstruct!>>
  do i = 1, n
    if (a(i) > 0.0) goto 10
    a(i) = 4.0
10  cycle
  end do

  ! A CYCLE that is not last is a real branch and is kept.
  ! CHECK:   <<DoConstruct!>> -> 29
  ! CHECK:     25 ^NonLabelDoStmt -> 28: do i = 1, n
  ! CHECK:     26 ^CycleStmt! -> 28: cycle
  ! CHECK:     27 ^AssignmentStmt: a(i) = 5.0
  ! CHECK:     28 ^EndDoStmt -> 25: end do
  ! CHECK:   <<End DoConstruct!>>
  do i = 1, n
    cycle
    a(i) = 5.0
  end do

  ! The lexical predecessor of the CYCLE is the last statement of the
  ! preceding construct, not the construct itself. Getting that wrong leaves
  ! the EndIfStmt pointing at the deleted CycleStmt, which shows up here as
  ! `<<IfConstruct>> -> 0'.
  ! CHECK:   <<DoConstruct>> -> 35
  ! CHECK:     29 ^NonLabelDoStmt -> 34: do i = 1, n
  ! CHECK:     <<IfConstruct>> -> 34
  ! CHECK:       30 ^IfThenStmt -> 34: if(a(i) > 0.0) then
  ! CHECK:       31 ^AssignmentStmt: a(i) = 7.0
  ! CHECK:       32 EndIfStmt: end if
  ! CHECK:     <<End IfConstruct>>
  ! CHECK:     34 EndDoStmt -> 29: end do
  ! CHECK:   <<End DoConstruct>>
  do i = 1, n
    if (a(i) > 0.0) then
      a(i) = 7.0
    end if
    cycle
  end do

  ! Same, with a nested DoConstruct as the preceding construct. Here a missing
  ! descent sends the inner loop's exit to 41 -- past the outer EndDoStmt.
  ! CHECK:   <<DoConstruct>> -> 41
  ! CHECK:     35 NonLabelDoStmt -> 40: do i = 1, n
  ! CHECK:     <<DoConstruct>> -> 40
  ! CHECK:       36 ^NonLabelDoStmt -> 38: do j = 1, n
  ! CHECK:       37 ^AssignmentStmt: a(j) = 8.0
  ! CHECK:       38 EndDoStmt -> 36: end do
  ! CHECK:     <<End DoConstruct>>
  ! CHECK:     40 EndDoStmt -> 35: end do
  ! CHECK:   <<End DoConstruct>>
  do i = 1, n
    do j = 1, n
      a(j) = 8.0
    end do
    cycle
  end do

  ! A trailing CYCLE preceded by an `if (cond) cycle'. The two rewrites have no
  ! strict ordering requirement here, because either way the surrounding DO
  ! comes out structured, which is the property that matters.
  !
  ! rewriteTrailingCycle first: the trailing CYCLE is deleted, which leaves the
  ! `if (cond) cycle' reaching the EndDoStmt, so rewriteIfGotos flips the
  ! condition and absorbs the body; both branches are gone.
  !
  ! rewriteIfGotos first: it flips the condition and splices the body -- which
  ! includes the trailing CYCLE -- into the IfConstruct. The CYCLE is then no
  ! longer last in the DO's own evaluation list, so it survives and the
  ! IfConstruct is unstructured. The DO around it is still structured.
  !
  ! Only the inner IfConstruct's classification and the statement indexes
  ! differ between the two, so match just the invariant part.
  ! CHECK:   <<DoConstruct>> -> {{[0-9]+}}
  ! CHECK:     {{[0-9]+}} {{\^?}}NonLabelDoStmt -> {{[0-9]+}}: do i = 1, n
  ! CHECK:     {{[0-9]+}} ^IfStmt [negate] -> {{[0-9]+}}: if(a(i) > 0.0) cycle
  ! CHECK:   <<End DoConstruct>>
  do i = 1, n
    if (a(i) > 0.0) cycle
    a(i) = 6.0
    cycle
  end do
end subroutine trailing_cycle
