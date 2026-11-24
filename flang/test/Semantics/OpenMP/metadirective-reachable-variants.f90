!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

! Only METADIRECTIVE replacements that can be selected are checked against the
! following loop.

! A higher-scored implicit NOTHING always wins, so the DO can never be selected.
subroutine static_winner(n, a)
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp& when(user={condition(score(10): .true.)}:) &
  !$omp& when(user={condition(score(5): .true.)}: do collapse(2)) &
  !$omp& otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! A dynamic winner leaves the DO selectable when its condition is false.
subroutine dynamic_winner(flag, n, a)
  logical :: flag
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp& when(user={condition(score(10): flag)}:) &
  !ERROR: This construct requires a nest of depth 2, but the associated nest is a nest of depth 1
  !BECAUSE: COLLAPSE clause was specified with argument 2
  !$omp& when(user={condition(score(5): .true.)}: do collapse(2)) &
  !$omp& otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! MATCH_ANY with an unmatched TARGET still scores below the NOTHING.
subroutine match_any_ranked(n, a)
  integer :: n, a(n), i
  !$omp parallel
    !$omp metadirective &
    !$omp& when(user={condition(score(100): .true.)}: nothing) &
    !$omp& when(construct={target, parallel}, &
    !$omp& implementation={extension(match_any)}: simd collapse(2)) &
    !$omp& otherwise(nothing)
    do i = 1, n
      a(i) = i
    end do
  !$omp end parallel
end subroutine

! No enclosing SIMD, so construct={simd} cannot match.
subroutine no_enclosing_simd(n, a)
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp& when(construct={simd}: simd collapse(2)) otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! The construct selector must match the enclosing constructs in order.
subroutine selector_order(n, a)
  integer :: n, a(n, n), i, j
  !$omp parallel do
  do i = 1, n
    !$omp metadirective &
    !$omp& when(construct={do, parallel}: simd collapse(2)) otherwise(nothing)
    do j = 1, n
      a(j, i) = j
    end do
    !$omp metadirective &
    !ERROR: This construct requires a nest of depth 2, but the associated nest is a nest of depth 1
    !BECAUSE: COLLAPSE clause was specified with argument 2
    !$omp& when(construct={parallel, do}: simd collapse(2)) otherwise(nothing)
    do j = 1, n
      a(j, i) = j
    end do
  end do
end subroutine

! TARGET hides the enclosing PARALLEL.
subroutine target_boundary(n, a)
  integer :: n, a(n), i
  !$omp parallel
    !$omp target
      !$omp metadirective &
      !$omp& when(construct={parallel, target}: simd collapse(2)) &
      !$omp& otherwise(nothing)
      do i = 1, n
        a(i) = i
      end do
    !$omp end target
  !$omp end parallel
end subroutine

! The DO selected for the first loop does not enclose the second METADIRECTIVE.
subroutine after_selected_loop(n, a)
  integer :: n, a(n), i
  !$omp metadirective when(implementation={vendor(llvm)}: do) otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
  !$omp metadirective when(construct={do}: simd collapse(2)) otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! The PARALLEL selected for the region ends at END METADIRECTIVE.
subroutine after_selected_region(n, a)
  integer :: n, a(n), i
  !$omp begin metadirective &
  !$omp& when(implementation={vendor(llvm)}: parallel) otherwise(nothing)
    a(1) = 1
  !$omp end metadirective
  !$omp metadirective &
  !$omp& when(construct={parallel}: simd collapse(2)) otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! The TILE can never be selected, so neither can the UNROLL it applies.
subroutine unreachable_apply()
  !$omp metadirective &
  !$omp& when(user={condition(score(10): .true.)}: nothing) &
  !$omp& when(user={condition(score(5): .true.)}: &
  !$omp& tile sizes(2) apply(grid: unroll)) &
  !$omp& otherwise(nothing)
end subroutine

! Replacements of a BEGIN METADIRECTIVE are ranked in the same way.
subroutine delimited_winner(n, a)
  integer :: n, a(n), i
  !$omp begin metadirective &
  !$omp& when(user={condition(score(10): .true.)}: nothing) &
  !$omp& when(user={condition(score(5): .true.)}: do collapse(2)) &
  !$omp& otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
  !$omp end metadirective
end subroutine

! The DO selected for the outer loop encloses the inner METADIRECTIVE, so its
! SIMD can be selected.
subroutine in_selected_loop(n, a)
  integer :: n, a(n, n), i, j
  !$omp metadirective when(implementation={vendor(llvm)}: do) otherwise(nothing)
  do i = 1, n
    !$omp metadirective &
    !ERROR: This construct requires a nest of depth 2, but the associated nest is a nest of depth 1
    !BECAUSE: COLLAPSE clause was specified with argument 2
    !$omp& when(construct={do}: simd collapse(2)) otherwise(nothing)
    do j = 1, n
      a(j, i) = i
    end do
  end do
end subroutine

! The selected PARALLEL and the enclosing TARGET both enclose the inner
! METADIRECTIVE.
subroutine in_selected_region(n, a)
  integer :: n, a(n, n), i, j
  !$omp target
    !$omp begin metadirective &
    !$omp& when(implementation={vendor(llvm)}: parallel) otherwise(nothing)
    do i = 1, n
      !$omp metadirective &
      !ERROR: This construct requires a nest of depth 2, but the associated nest is a nest of depth 1
      !BECAUSE: COLLAPSE clause was specified with argument 2
      !$omp& when(construct={target, parallel}: simd collapse(2)) &
      !$omp& otherwise(nothing)
      do j = 1, n
        a(j, i) = i
      end do
    end do
    !$omp end metadirective
  !$omp end target
end subroutine
