!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51

! Static applicability of loop-associated METADIRECTIVE variants

! device={kind(nohost)} cannot match during host compilation so semantic check is skipped
! for this variant.
subroutine f01(n, a)
  integer :: n, a(n, n), i, j
  !$omp metadirective when(device={kind(nohost)}: do collapse(3)) default(nothing)
  do i = 1, n
    do j = 1, n
      a(j, i) = i
    end do
  end do
end subroutine

subroutine f02(n, a)
  integer :: n, a(n, n), i, j
  !ERROR: This construct requires a perfect nest of depth 3, but the associated nest is a perfect nest of depth 2
  !BECAUSE: COLLAPSE clause was specified with argument 3
  !$omp metadirective when(implementation={vendor(llvm)}: do collapse(3)) default(nothing)
  do i = 1, n
    do j = 1, n
      a(j, i) = i
    end do
  end do
end subroutine

! Variant not skipped since a non-constant user condition may be selected at run time.
subroutine f03(n, a, flag)
  integer :: n, a(n, n), i, j
  logical :: flag
  !ERROR: This construct requires a perfect nest of depth 3, but the associated nest is a perfect nest of depth 2
  !BECAUSE: COLLAPSE clause was specified with argument 3
  !$omp metadirective when(user={condition(flag)}: do collapse(3)) default(nothing)
  do i = 1, n
    do j = 1, n
      a(j, i) = i
    end do
  end do
end subroutine

! A dead WHEN clause must not suppress the unguarded DEFAULT variant.
subroutine f04(n, a)
  integer :: n, a(n, n), i, j
  !ERROR: This construct requires a perfect nest of depth 3, but the associated nest is a perfect nest of depth 2
  !BECAUSE: COLLAPSE clause was specified with argument 3
  !$omp metadirective when(device={kind(nohost)}: nothing) default(do collapse(3))
  do i = 1, n
    do j = 1, n
      a(j, i) = i
    end do
  end do
end subroutine

! A user condition that folds to a compile-time false makes the variant
! unselectable, so its loop is skipped and DEFAULT applies.
subroutine f05(n, a)
  integer :: n, a(n, n), i, j
  logical, parameter :: use_variant = .false.
  !$omp metadirective when(user={condition(use_variant)}: do collapse(3)) default(nothing)
  do i = 1, n
    do j = 1, n
      a(j, i) = i
    end do
  end do
end subroutine

! A user condition that folds to a compile-time true keeps the variant, so its
! loop is still checked.
subroutine f06(n, a)
  integer :: n, a(n, n), i, j
  logical, parameter :: use_variant = .true.
  !ERROR: This construct requires a perfect nest of depth 3, but the associated nest is a perfect nest of depth 2
  !BECAUSE: COLLAPSE clause was specified with argument 3
  !$omp metadirective when(user={condition(use_variant)}: do collapse(3)) default(nothing)
  do i = 1, n
    do j = 1, n
      a(j, i) = i
    end do
  end do
end subroutine

! An unknown implementation VENDOR never matches, so the variant is skipped.
subroutine f07(n, a)
  integer :: n, a(n, n), i, j
  !$omp metadirective when(implementation={vendor(bogus_vendor)}: do collapse(3)) default(nothing)
  do i = 1, n
    do j = 1, n
      a(j, i) = i
    end do
  end do
end subroutine

! An unknown device ARCH never matches, so the variant is skipped.
subroutine f08(n, a)
  integer :: n, a(n, n), i, j
  !$omp metadirective when(device={arch(bogus_arch)}: do collapse(3)) default(nothing)
  do i = 1, n
    do j = 1, n
      a(j, i) = i
    end do
  end do
end subroutine

! MATCH_NONE is satisfied by the unmatched (invalid) vendor, so the variant
! stays selectable and its loop must still be checked.
subroutine f09(n, a)
  integer :: n, a(n, n), i, j
  !ERROR: This construct requires a perfect nest of depth 3, but the associated nest is a perfect nest of depth 2
  !BECAUSE: COLLAPSE clause was specified with argument 3
  !$omp metadirective when(implementation={vendor(bogus_vendor), extension(match_none)}: do collapse(3)) default(nothing)
  do i = 1, n
    do j = 1, n
      a(j, i) = i
    end do
  end do
end subroutine

! MATCH_ANY is satisfied by a compile-time-true user condition, so its
! loop-associated variant still requires a loop.
subroutine f10()
  logical, parameter :: use_variant = .true.
  !ERROR: This construct should contain a DO-loop or a loop-nest-generating construct
  !$omp metadirective when(user={condition(use_variant)}, implementation={extension(match_any)}: do) default(nothing)
end subroutine

! MATCH_NONE is not satisfied when the user condition is true, so its
! loop-associated variant cannot be selected and needs no loop.
subroutine f11()
  logical, parameter :: use_variant = .true.
  !$omp metadirective when(user={condition(use_variant)}, implementation={extension(match_none)}: do) default(nothing)
end subroutine
