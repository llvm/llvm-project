! RUN: not %flang_fc1 -fopenmp -fopenmp-version=51 -fsyntax-only %s 2>&1 | \
! RUN:   FileCheck %s --implicit-check-not=error:

! CHECK: error: Semantic errors in

! Both paths lack TARGET and contain PARALLEL, but MATCH_ANY scores their
! first PARALLEL at different positions. PARALLEL -> PARALLEL scores 2 and
! loses the tie with NOTHING. TEAMS -> PARALLEL scores 3 and selects SIMD.
! Merging these paths must not discard the reachable COLLAPSE diagnostic.
subroutine parallel_first(flag, n, a)
  logical :: flag
  integer :: n, a(n), i
  !$omp begin metadirective &
  !$omp& when(user={condition(flag)}: parallel) default(teams)
    !$omp parallel
      !$omp metadirective &
      !$omp& when(user={condition(score(1): .true.)}: nothing) &
      !$omp& when(construct={target, parallel}, &
! CHECK: :[[@LINE+3]]:{{[0-9]+}}: error: This construct requires
! CHECK-SAME: a nest of depth 2, but the associated nest is a nest of depth 1
! CHECK: because: COLLAPSE clause was specified with argument 2
      !$omp& implementation={extension(match_any)}: simd collapse(2)) &
      !$omp& default(nothing)
      do i = 1, n
        a(i) = i
      end do
    !$omp end parallel
  !$omp end metadirective
end subroutine

! Reversing the path order must leave the same replacement reachable.
subroutine teams_first(flag, n, a)
  logical :: flag
  integer :: n, a(n), i
  !$omp begin metadirective &
  !$omp& when(user={condition(flag)}: teams) default(parallel)
    !$omp parallel
      !$omp metadirective &
      !$omp& when(user={condition(score(1): .true.)}: nothing) &
      !$omp& when(construct={target, parallel}, &
! CHECK: :[[@LINE+3]]:{{[0-9]+}}: error: This construct requires
! CHECK-SAME: a nest of depth 2, but the associated nest is a nest of depth 1
! CHECK: because: COLLAPSE clause was specified with argument 2
      !$omp& implementation={extension(match_any)}: simd collapse(2)) &
      !$omp& default(nothing)
      do i = 1, n
        a(i) = i
      end do
    !$omp end parallel
  !$omp end metadirective
end subroutine

! A score-4 NOTHING wins on both paths, so retaining both contexts must not
! make the lower-scored SIMD replacement reachable.
subroutine nothing_wins(flag, n, a)
  logical :: flag
  integer :: n, a(n), i
  !$omp begin metadirective &
  !$omp& when(user={condition(flag)}: parallel) default(teams)
    !$omp parallel
      !$omp metadirective &
      !$omp& when(user={condition(score(3): .true.)}: nothing) &
      !$omp& when(construct={target, parallel}, &
      !$omp& implementation={extension(match_any)}: simd collapse(2)) &
      !$omp& default(nothing)
      do i = 1, n
        a(i) = i
      end do
    !$omp end parallel
  !$omp end metadirective
end subroutine
