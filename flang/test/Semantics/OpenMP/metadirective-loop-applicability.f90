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
  !ERROR: This construct requires a nest of depth 3, but the associated nest is a nest of depth 2
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
  !ERROR: This construct requires a nest of depth 3, but the associated nest is a nest of depth 2
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
  !ERROR: This construct requires a nest of depth 3, but the associated nest is a nest of depth 2
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
  !ERROR: This construct requires a nest of depth 3, but the associated nest is a nest of depth 2
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
  !ERROR: This construct requires a nest of depth 3, but the associated nest is a nest of depth 2
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

! A higher-scored static implicit NOTHING makes the lower-scored loop variant
! unreachable, so the latter's loop requirements are not checked.
subroutine f12(n, a)
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp& when(user={condition(score(10): .true.)}:) &
  !$omp& when(user={condition(score(5): .true.)}: do collapse(2)) &
  !$omp& default(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! A dynamic implicit NOTHING leaves the lower-scored loop variant reachable
! when its condition is false, so that variant's requirements are checked.
subroutine f13(flag, n, a)
  logical :: flag
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp& when(user={condition(score(10): flag)}:) &
  !ERROR: This construct requires a nest of depth 2, but the associated nest is a nest of depth 1
  !BECAUSE: COLLAPSE clause was specified with argument 2
  !$omp& when(user={condition(score(5): .true.)}: do collapse(2)) &
  !$omp& default(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! A nested construct selector observes the loop directive selected by a
! standalone metadirective.
subroutine f14(n, a)
  integer :: n, a(n, n), i, j
  !$omp metadirective when(implementation={vendor(llvm)}: do) default(nothing)
  do i = 1, n
    !ERROR: This construct requires a nest of depth 2, but the associated nest is a nest of depth 1
    !BECAUSE: COLLAPSE clause was specified with argument 2
    !$omp metadirective when(construct={do}: simd collapse(2)) default(nothing)
    do j = 1, n
      a(j, i) = i
    end do
  end do
end subroutine

! Actual and selected contexts are both visible inside a begin/end
! metadirective.
subroutine f15(n, a)
  integer :: n, a(n, n), i, j
  !$omp target
    !$omp begin metadirective &
    !$omp& when(implementation={vendor(llvm)}: parallel) default(nothing)
    do i = 1, n
      !ERROR: This construct requires a nest of depth 2, but the associated nest is a nest of depth 1
      !BECAUSE: COLLAPSE clause was specified with argument 2
      !$omp metadirective when(construct={target, parallel}: simd collapse(2)) default(nothing)
      do j = 1, n
        a(j, i) = i
      end do
    end do
    !$omp end metadirective
  !$omp end target
end subroutine

! Mutually exclusive selected contexts are not flattened together. No path has
! both TARGET and PARALLEL, so the first inner loop variant is unreachable.
! The second inner variant verifies that the dynamic TARGET path is retained.
subroutine f16(flag, n, a)
  logical :: flag
  integer :: n, a(n), i
  !$omp begin metadirective &
  !$omp& when(user={condition(flag)}: target) default(parallel)
    !$omp metadirective &
    !$omp& when(construct={target, parallel}: simd collapse(2)) &
    !$omp& default(nothing)
    do i = 1, n
      a(i) = i
    end do

    !ERROR: This construct requires a nest of depth 2, but the associated nest is a nest of depth 1
    !BECAUSE: COLLAPSE clause was specified with argument 2
    !$omp metadirective when(construct={target}: simd collapse(2)) default(nothing)
    do i = 1, n
      a(i) = i
    end do
  !$omp end metadirective
end subroutine

! A non-NOTHING fallback contributes its selected context to nested matching.
subroutine f17(flag, n, a)
  logical :: flag
  integer :: n, a(n), i
  !$omp begin metadirective &
  !$omp& when(user={condition(flag)}: target) default(parallel)
    !ERROR: This construct requires a nest of depth 2, but the associated nest is a nest of depth 1
    !BECAUSE: COLLAPSE clause was specified with argument 2
    !$omp metadirective when(construct={parallel}: simd collapse(2)) default(nothing)
    do i = 1, n
      a(i) = i
    end do
  !$omp end metadirective
end subroutine

! A selected context ends with its associated loop and does not affect a
! subsequent metadirective.
subroutine f18(n, a)
  integer :: n, a(n), i
  !$omp metadirective when(implementation={vendor(llvm)}: do) default(nothing)
  do i = 1, n
    a(i) = i
  end do

  !$omp metadirective when(construct={do}: simd collapse(2)) default(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! A selected context ends at END METADIRECTIVE and does not affect a
! subsequent metadirective.
subroutine f19(n, a)
  integer :: n, a(n), i
  !$omp begin metadirective &
  !$omp& when(implementation={vendor(llvm)}: parallel) default(nothing)
    a(1) = 1
  !$omp end metadirective

  !$omp metadirective when(construct={parallel}: simd collapse(2)) default(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! Unsupported selectors conservatively retain their explicit replacement.
subroutine f20(n, a)
  integer :: n, a(n), i
  !$omp metadirective &
  !ERROR: This construct requires a nest of depth 2, but the associated nest is a nest of depth 1
  !BECAUSE: COLLAPSE clause was specified with argument 2
  !$omp& when(target_device={kind(host)}: do collapse(2)) default(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! Unsupported selectors also retain the DEFAULT replacement.
subroutine f21(n, a)
  integer :: n, a(n), i
  !$omp metadirective &
  !ERROR: This construct requires a nest of depth 2, but the associated nest is a nest of depth 1
  !BECAUSE: COLLAPSE clause was specified with argument 2
  !$omp& when(target_device={kind(host)}: nothing) default(do collapse(2))
  do i = 1, n
    a(i) = i
  end do
end subroutine

! Recovery retains replacements from siblings of an unsupported selector.
subroutine f22(n, a)
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp& when(target_device={kind(host)}: nothing) &
  !ERROR: This construct requires a nest of depth 2, but the associated nest is a nest of depth 1
  !BECAUSE: COLLAPSE clause was specified with argument 2
  !$omp& when(implementation={vendor(llvm)}: do collapse(2)) &
  !$omp& default(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! A selected composite directive contributes each of its leaf constructs.
subroutine f23(n, a)
  integer :: n, a(n, n), i, j
  !$omp metadirective &
  !$omp& when(implementation={vendor(llvm)}: parallel do) default(nothing)
  do i = 1, n
    !$omp metadirective &
    !ERROR: This construct requires a nest of depth 2, but the associated nest is a nest of depth 1
    !BECAUSE: COLLAPSE clause was specified with argument 2
    !$omp& when(construct={parallel, do}: simd collapse(2)) default(nothing)
    do j = 1, n
      a(j, i) = i
    end do
  end do
end subroutine

! An actual TARGET hides enclosing constructs from a nested selector.
subroutine f24(n, a)
  integer :: n, a(n), i
  !$omp parallel
    !$omp target
      !$omp metadirective &
      !$omp& when(construct={parallel, target}: &
      !$omp& simd collapse(2)) default(nothing)
      do i = 1, n
        a(i) = i
      end do
    !$omp end target
  !$omp end parallel
end subroutine

! A selected TARGET also hides enclosing constructs from a nested selector.
subroutine f25(n, a)
  integer :: n, a(n), i
  !$omp parallel
    !$omp begin metadirective &
    !$omp& when(implementation={vendor(llvm)}: target) &
    !$omp& default(nothing)
      !$omp metadirective &
      !$omp& when(construct={parallel, target}: &
      !$omp& simd collapse(2)) default(nothing)
      do i = 1, n
        a(i) = i
      end do
    !$omp end metadirective
  !$omp end parallel
end subroutine
