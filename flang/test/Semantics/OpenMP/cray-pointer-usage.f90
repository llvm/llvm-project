!RUN: %python %S/../test_errors.py %s %flang -fopenmp
subroutine test_cray_pointer_usage
  implicit none
  integer :: i
  real(8) :: var(*), pointee(2)
  pointer(ivar, var)
  ! ERROR: Cray Pointee 'var' may not appear in LINEAR clause
  ! ERROR: The list item 'var' specified without the REF 'linear-modifier' must be of INTEGER type
  ! ERROR: The list item `var` must be a dummy argument
  !$omp declare simd linear(var)

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

  ! ERROR: Cray Pointee 'var' may not appear in SHARED clause, use Cray Pointer 'ivar' instead
  !$omp parallel num_threads(2) default(none) shared(var)
    print *, var(1)
  !$omp end parallel

  ! ERROR: Cray Pointee 'var' may not appear in LASTPRIVATE clause, use Cray Pointer 'ivar' instead
  !$omp do lastprivate(var)
    do i = 1, 10
      print *, var(1)
    end do
  !$omp end do

  !$omp parallel num_threads(2) default(none) firstprivate(ivar)
    print *, var(1)
  !$omp end parallel

  !$omp parallel num_threads(2) default(private) shared(ivar)
    print *, var(1)
  !$omp end parallel
end subroutine test_cray_pointer_usage
