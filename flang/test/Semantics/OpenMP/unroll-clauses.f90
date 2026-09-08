! Testing the Semantics of the FULL and PARTIAL clauses on the UNROLL directive

!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51

subroutine unroll_clauses
  implicit none
  integer, parameter :: n = 8
  integer :: i
  integer :: v(n)

  !ERROR: FULL and PARTIAL clauses are mutually exclusive and may not appear on the same UNROLL directive
  !$omp unroll full partial(2)
  do i = 1, n
    v(i) = i
  end do
  !$omp end unroll

  !ERROR: At most one FULL clause can appear on UNROLL directive
  !$omp unroll full full
  do i = 1, n
    v(i) = i
  end do
  !$omp end unroll

  ! Each clause on its own is accepted.
  !$omp unroll full
  do i = 1, n
    v(i) = i
  end do
  !$omp end unroll

  !$omp unroll partial(2)
  do i = 1, n
    v(i) = i
  end do
  !$omp end unroll
end subroutine

subroutine unroll_full_trip_count(m, s)
  implicit none
  integer, parameter :: n = 8
  integer :: m, s, i
  integer :: v(n)

  ! A fully unrolled loop must have a trip count known at compile time.
  !ERROR: The loop associated with an UNROLL directive with a FULL clause must have a constant trip count
  !$omp unroll full
  do i = 1, m
    v(1) = i
  end do
  !$omp end unroll

  !ERROR: The loop associated with an UNROLL directive with a FULL clause must have a constant trip count
  !$omp unroll full
  do i = 1, n, s
    v(1) = i
  end do
  !$omp end unroll

  ! Constant bounds are fine, including named constants.
  !$omp unroll full
  do i = 1, n
    v(i) = i
  end do
  !$omp end unroll

  ! PARTIAL places no such requirement on the loop.
  !$omp unroll partial(2)
  do i = 1, m
    v(1) = i
  end do
  !$omp end unroll
end subroutine
