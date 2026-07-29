! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=60

! Only directives that have the "pure" property are permitted to appear in a
! Fortran PURE procedure. These are: metadirective, assume, assumes, nothing,
! error, and the loop-transforming directives (tile, unroll, reverse,
! interchange, fuse, split, stripe).
! (The SPLIT directive is omitted below only because its required COUNTS clause
! is not yet parsed; it is still marked pure.)

module m
contains
  pure subroutine pure_ok(a, n)
    !$omp assumes no_openmp
    integer, intent(in) :: n
    integer, intent(inout) :: a(n)
    integer :: i, j
    !$omp nothing
    !$omp assume no_openmp
    !$omp end assume
    !$omp metadirective when(user={condition(.true.)}: nothing)
    !$omp error at(execution) severity(warning) message("ok")
    !$omp tile sizes(4)
    do i = 1, n
      a(i) = a(i) + 1
    end do
    !$omp unroll partial(2)
    do i = 1, n
      a(i) = a(i) * 2
    end do
    !$omp reverse
    do i = 1, n
      a(i) = a(i) - 1
    end do
    !$omp interchange
    do i = 1, n
      do j = 1, n
        a(i) = a(i) + j
      end do
    end do
    !$omp stripe sizes(4)
    do i = 1, n
      a(i) = a(i) + 1
    end do
    !$omp fuse
    do i = 1, n
      a(i) = a(i) + 1
    end do
    do i = 1, n
      a(i) = a(i) + 2
    end do
  end subroutine

  pure subroutine pure_bad(a, n, r)
    integer, intent(in) :: n
    integer, intent(inout) :: a(n)
    integer, intent(out) :: r
    integer :: i
    !ERROR: The OpenMP directive 'PARALLEL' is not allowed in a PURE procedure
    !$omp parallel
    !$omp end parallel
    !ERROR: The OpenMP directive 'BARRIER' is not allowed in a PURE procedure
    !$omp barrier
    !ERROR: The OpenMP directive 'ATOMIC' is not allowed in a PURE procedure
    !$omp atomic
    r = r + 1
    !ERROR: The OpenMP directive 'PARALLEL DO' is not allowed in a PURE procedure
    !$omp parallel do
    do i = 1, n
      a(i) = a(i) + 1
    end do
    !$omp end parallel do
    !ERROR: The OpenMP directive 'TASK' is not allowed in a PURE procedure
    !$omp task
    !$omp end task
  end subroutine
end module
