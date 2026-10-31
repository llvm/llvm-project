!RUN: %python %S/../test_errors.py %s %flang -fopenmp
! Negative tests for default(none)

! Check that the same error is not displayed for every use of a symbol.
subroutine repeated_error()
  real :: B

  !$omp parallel default(none)
    ! ERROR: The DEFAULT(NONE) clause requires that 'b' must be listed in a data-sharing attribute clause
    B = B + 1.0 + B
  !$omp end parallel

  !$omp parallel default(none)
  !$omp critical
    ! ERROR: The DEFAULT(NONE) clause requires that 'b' must be listed in a data-sharing attribute clause
    B = B + 2.0 + B
    !$omp parallel default(none)
      ! ERROR: The DEFAULT(NONE) clause requires that 'b' must be listed in a data-sharing attribute clause
      B = B + 3.0 + B
    !$omp end parallel
  !$omp end critical
  !$omp end parallel
end subroutine
