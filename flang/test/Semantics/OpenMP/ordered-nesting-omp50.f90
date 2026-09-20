! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=50
! OpenMP 5.0: an ORDERED clause requires perfect nesting, whether or not the
! clause has an argument. This is stricter than 5.1/5.2, where only ORDERED(n)
! (with an argument) requires perfect nesting (see ordered-nesting-omp51.f90).

! Bare ORDERED with COLLAPSE(2): the two associated loops must be perfectly
! nested in 5.0, so intervening code between them is an error.
subroutine bare_ordered_collapse_imperfect
  integer i, j

  !ERROR: This construct requires a perfect nest of depth 2, but the associated nest is a perfect nest of depth 1
  !BECAUSE: COLLAPSE clause was specified with argument 2
  !$omp do collapse(2) ordered
  do i = 1, 10
    !BECAUSE: This code prevents perfect nesting
    print *, i
    do j = 1, 10
      print *, i, j
    end do
  end do
  !$omp end do
end subroutine

! Bare ORDERED with COLLAPSE(2), perfectly nested: valid.
subroutine bare_ordered_collapse_perfect
  integer i, j

  !$omp do collapse(2) ordered
  do i = 1, 10
    do j = 1, 10
      print *, i, j
    end do
  end do
  !$omp end do
end subroutine

! Bare ORDERED with no COLLAPSE: only one loop is associated (depth 1), so
! perfect nesting is trivially satisfied and intervening code is allowed.
subroutine bare_ordered_no_collapse
  integer i, j

  !$omp do ordered
  do i = 1, 10
    print *, i
    do j = 1, 10
      print *, i, j
    end do
  end do
  !$omp end do
end subroutine
