!RUN: %python %S/../test_errors.py %s %flang -fopenmp
subroutine test_cray_pointer_usage
  implicit none
  real(8) :: var(*), pointee(2)
  pointer(ivar, var)

  pointee = 42.0
  ivar = loc(pointee)

  !$omp parallel num_threads(2) default(none)
    ! ERROR: The DEFAULT(NONE) clause requires that the Cray Pointer 'ivar' must be listed in a data-sharing attribute clause
    print *, var(1)
  !$omp end parallel

  ! ERROR: Cray Pointee 'var' may not appear in PRIVATE clause, use Cray Pointer 'ivar' instead
  !$omp parallel num_threads(2) default(none) private(var)
    print *, var(1)
  !$omp end parallel

  !$omp parallel num_threads(2) default(none) firstprivate(ivar)
    print *, var(1)
  !$omp end parallel

  !$omp parallel num_threads(2) default(private) shared(ivar)
    print *, var(1)
  !$omp end parallel
end subroutine test_cray_pointer_usage
