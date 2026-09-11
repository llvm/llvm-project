! RUN: %flang_fc1 -fopenmp -fdebug-dump-pft %s 2>&1 | FileCheck %s

! A CYCLE that is the last statement of the body of its own DO is deleted, and
! the lexical predecessor it is unlinked from is the last *statement* reachable
! from the preceding evaluation, found by descending through nested evaluation
! lists.
!
! Most constructs end with an End<construct>Stmt, so that descent stops after
! one step. An OpenMP loop construct does not: exitConstructOrDirective skips
! appending the ContinueStmt exit target for it, so the OpenMPConstruct's
! evaluation list ends with the associated DoConstruct, and reaching the
! EndDoStmt takes two steps. Descending only one step leaves that EndDoStmt
! pointing at the deleted CycleStmt, which surfaces below as the inner loop
! getting `-> 8' -- the EndSubroutineStmt, past the outer EndDoStmt.

! CHECK: 1 Subroutine s
subroutine s(a, n)
  integer :: n, i, j
  real :: a(n)

  ! CHECK:   <<DoConstruct>> -> 8
  ! CHECK:     1 NonLabelDoStmt -> 7: do i = 1, n
  ! CHECK:     2 <<^OpenMPConstruct>>
  ! CHECK:       <<DoConstruct>> -> 7
  ! CHECK:         3 NonLabelDoStmt -> 5: do j = 1, n
  ! CHECK:         4 ^AssignmentStmt: a(j) = 1.0
  ! CHECK:         5 EndDoStmt -> 3: end do
  ! CHECK:       <<End DoConstruct>>
  ! CHECK:     <<End OpenMPConstruct>>
  ! CHECK:     7 EndDoStmt -> 1: end do
  ! CHECK:   <<End DoConstruct>>
  do i = 1, n
    !$omp do
    do j = 1, n
      a(j) = 1.0
    end do
    cycle
  end do
  ! CHECK:   8 EndSubroutineStmt
end subroutine s
