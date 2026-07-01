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
