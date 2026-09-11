! RUN: not %flang_fc1 -fopenmp -fopenmp-version=51 -fsyntax-only %s 2>&1 | \
! RUN:   FileCheck %s --implicit-check-not=error:

! CHECK: error: Semantic errors in

! Matching the inner PARALLEL gives the construct candidate a score of 3.
subroutine repeated_parallel(n)
  integer :: n, i
  !$omp parallel
    !$omp parallel
      !$omp metadirective &
      !$omp& when(user={condition(score(1): .true.)}: nothing) &
! CHECK: :[[@LINE+3]]:{{[0-9]+}}: error: This construct requires
! CHECK-SAME: a nest of depth 2, but the associated nest is a nest of depth 1
! CHECK: because: COLLAPSE clause was specified with argument 2
      !$omp& when(construct={parallel}: simd collapse(2)) default(nothing)
      do i = 1, n
      end do
    !$omp end parallel
  !$omp end parallel
end subroutine

! Both paths have the same first PARALLEL match, but different last matches.
! Merging them must not lose the higher-scoring PARALLEL -> PARALLEL path.
subroutine repeated_parallel_paths(flag, n)
  logical :: flag
  integer :: n, i
  !$omp parallel
    !$omp begin metadirective &
    !$omp& when(user={condition(flag)}: teams) default(parallel)
      !$omp metadirective &
      !$omp& when(user={condition(score(1): .true.)}: nothing) &
! CHECK: :[[@LINE+3]]:{{[0-9]+}}: error: This construct requires
! CHECK-SAME: a nest of depth 2, but the associated nest is a nest of depth 1
! CHECK: because: COLLAPSE clause was specified with argument 2
      !$omp& when(construct={parallel}: simd collapse(2)) default(nothing)
      do i = 1, n
      end do
    !$omp end metadirective
  !$omp end parallel
end subroutine

! Before DO is appended, neither path fully matches {parallel, do}. The best
! PARALLEL prefix must survive merging to obtain the correct score afterwards.
subroutine repeated_parallel_prefix(flag, n)
  logical :: flag
  integer :: n, i, j
  !$omp parallel
    !$omp begin metadirective &
    !$omp& when(user={condition(flag)}: teams) default(parallel)
      !$omp do
      do i = 1, n
        !$omp metadirective &
        !$omp& when(user={condition(score(5): .true.)}: nothing) &
! CHECK: :[[@LINE+3]]:{{[0-9]+}}: error: This construct requires
! CHECK-SAME: a nest of depth 2, but the associated nest is a nest of depth 1
! CHECK: because: COLLAPSE clause was specified with argument 2
        !$omp& when(construct={parallel, do}: simd collapse(2)) default(nothing)
        do j = 1, n
        end do
      end do
      !$omp end do
    !$omp end metadirective
  !$omp end parallel
end subroutine

! MATCH_NONE with an unknown vendor remains applicable during ranking.
subroutine match_none_unknown_vendor(n)
  integer :: n, i
  !$omp metadirective &
  !$omp& when(implementation={vendor(bogus_vendor), &
! CHECK: :[[@LINE+3]]:{{[0-9]+}}: error: This construct requires
! CHECK-SAME: a nest of depth 2, but the associated nest is a nest of depth 1
! CHECK: because: COLLAPSE clause was specified with argument 2
  !$omp& extension(match_none)}: simd collapse(2)) &
  !$omp& when(user={condition(.true.)}: nothing) default(nothing)
  do i = 1, n
  end do
end subroutine

! A higher-scored dynamic implicit NOTHING leaves a reachable path without DO.
subroutine omitted_score(flag, n)
  logical :: flag
  integer :: n, i, j
  !$omp metadirective &
  !$omp& when(user={condition(score(10): flag)}:) &
  !$omp& when(user={condition(score(5): .true.)}: do) default(nothing)
  do i = 1, n
    !$omp metadirective when(construct={do}: nothing) &
! CHECK: :[[@LINE+3]]:{{[0-9]+}}: error: This construct requires
! CHECK-SAME: a nest of depth 2, but the associated nest is a nest of depth 1
! CHECK: because: COLLAPSE clause was specified with argument 2
    !$omp& default(simd collapse(2))
    do j = 1, n
    end do
  end do
end subroutine

! The outer choice and the inner selector produce PARALLEL -> DO first,
! then DO -> PARALLEL. Both paths have the same length and trait presence,
! but only the second matches the innermost ordered construct selector.
! Merging by trait presence alone would discard its COLLAPSE diagnostic.
subroutine ordered_paths(flag, n, a)
  logical :: flag
  integer :: n, a(n, n, n), i, j, k
  !$omp begin metadirective &
  !$omp& when(user={condition(flag)}: parallel) default(do)
  do i = 1, n
    !$omp begin metadirective &
    !$omp& when(construct={do}: parallel) default(do)
    do j = 1, n
      !$omp metadirective &
! CHECK: :[[@LINE+3]]:{{[0-9]+}}: error: This construct requires
! CHECK-SAME: a nest of depth 2, but the associated nest is a nest of depth 1
! CHECK: because: COLLAPSE clause was specified with argument 2
      !$omp& when(construct={do, parallel}: simd collapse(2)) default(nothing)
      do k = 1, n
        a(k, j, i) = k
      end do
    end do
    !$omp end metadirective
  end do
  !$omp end metadirective
end subroutine

! Consuming the enclosing pending group must not invalidate scope boundaries.
! Neither inner variant can associate with a loop outside the selected region.
subroutine escaped_region(flag, n)
  logical :: flag
  integer :: n, i
  !$omp begin metadirective default(parallel)
    continue
    !$omp metadirective &
! CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: This construct should contain
! CHECK-SAME: a DO-loop or a loop-nest-generating construct
    !$omp& when(user={condition(flag)}: simd) &
! CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: This construct should contain
! CHECK-SAME: a DO-loop or a loop-nest-generating construct
    !$omp& default(do)
  !$omp end metadirective
  do i = 1, n
  end do
end subroutine

! Consuming pending groups at a loop preserves every enclosing scope boundary,
! including those of nested selected regions.
subroutine escaped_nested_region_after_loop(flag, n)
  logical :: flag
  integer :: n, i
  !$omp begin metadirective default(parallel)
    !$omp begin metadirective default(parallel)
      do i = 1, n
      end do
      !$omp metadirective &
! CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: This construct should contain
! CHECK-SAME: a DO-loop or a loop-nest-generating construct
      !$omp& when(user={condition(flag)}: simd) &
! CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: This construct should contain
! CHECK-SAME: a DO-loop or a loop-nest-generating construct
      !$omp& default(do)
    !$omp end metadirective
    do i = 1, n
    end do
  !$omp end metadirective
end subroutine

! A loop within the selected region still satisfies both inner variants.
subroutine associated_loop_in_region(flag, n)
  logical :: flag
  integer :: n, i
  !$omp begin metadirective default(parallel)
    continue
    !$omp metadirective when(user={condition(flag)}: simd) default(do)
    do i = 1, n
    end do
  !$omp end metadirective
end subroutine

! PARALLEL -> DO alone must not match the reversed selector.
subroutine reversed_selector(n, a)
  integer :: n, a(n, n), i, j
  !$omp parallel do
  do i = 1, n
    !$omp metadirective &
    !$omp& when(construct={do, parallel}: simd collapse(2)) default(nothing)
    do j = 1, n
      a(j, i) = j
    end do
  end do
  !$omp end parallel do
end subroutine

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

! A bare SIMD selector must not match an empty construct context.
subroutine absent_simd(n, a)
  integer :: n, a(n), i
  !$omp metadirective when(construct={simd}: nothing) &
! CHECK: :[[@LINE+3]]:{{[0-9]+}}: error: This construct requires
! CHECK-SAME: a nest of depth 2, but the associated nest is a nest of depth 1
! CHECK: because: COLLAPSE clause was specified with argument 2
  !$omp& default(simd collapse(2))
  do i = 1, n
    a(i) = i
  end do
end subroutine

! The same absent trait must keep an invalid WHEN replacement unreachable.
subroutine unreachable_simd(n, a)
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp& when(construct={simd}: simd collapse(2)) default(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! A selected SIMD replacement supplies the trait to its associated loop.
subroutine present_simd(n, a)
  integer :: n, a(n, n), i, j
  !$omp metadirective default(simd)
  do i = 1, n
    !$omp metadirective when(construct={simd}: nothing) &
    !$omp& default(simd collapse(2))
    do j = 1, n
      a(i, j) = i + j
    end do
  end do
end subroutine

! DO and SIMD paths must remain distinct even when SIMD is the only selector
! in this program unit. Keeping only the first path would lose the error.
subroutine do_first(flag, n, a)
  logical :: flag
  integer :: n, a(n, n), i, j
  !$omp metadirective &
  !$omp& when(user={condition(flag)}: do) default(simd)
  do i = 1, n
    !$omp metadirective &
! CHECK: :[[@LINE+3]]:{{[0-9]+}}: error: This construct requires
! CHECK-SAME: a nest of depth 2, but the associated nest is a nest of depth 1
! CHECK: because: COLLAPSE clause was specified with argument 2
    !$omp& when(construct={simd}: simd collapse(2)) default(nothing)
    do j = 1, n
      a(i, j) = i + j
    end do
  end do
end subroutine

! Reversing the alternatives must also retain the path without SIMD.
subroutine simd_first(flag, n, a)
  logical :: flag
  integer :: n, a(n, n), i, j
  !$omp metadirective &
  !$omp& when(user={condition(flag)}: simd) default(do)
  do i = 1, n
    !$omp metadirective when(construct={simd}: nothing) &
! CHECK: :[[@LINE+3]]:{{[0-9]+}}: error: This construct requires
! CHECK-SAME: a nest of depth 2, but the associated nest is a nest of depth 1
! CHECK: because: COLLAPSE clause was specified with argument 2
    !$omp& default(simd collapse(2))
    do j = 1, n
      a(i, j) = i + j
    end do
  end do
end subroutine
