!RUN: %python %S/../test_errors.py %s %flang -fopenmp
subroutine test_crayptr
  implicit none
  integer :: I
  real*8 var(*)
  pointer(ivar,var)
  real*8 pointee(8)

  pointee(1) = 42.0
  ivar = loc(pointee)

  !$omp parallel num_threads(2) default(none) shared(ivar)
    print *, var(1)
  !$omp end parallel

  !$omp parallel num_threads(2) default(none) private(ivar)
    print *, var(1)
  !$omp end parallel

  !$omp parallel num_threads(2) default(private) shared(ivar)
    print *, var(1)
  !$omp end parallel

  !$omp parallel num_threads(2) default(firstprivate) shared(ivar)
    print *, var(1)
  !$omp end parallel

end subroutine test_crayptr

