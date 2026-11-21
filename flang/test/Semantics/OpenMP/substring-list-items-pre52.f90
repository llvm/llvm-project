! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=45
! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=50
! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=51

! Before 5.2 there is no general substring exclusion. Data-sharing clauses
! still prohibit parts of variables through their own restrictions.
subroutine data_sharing(c)
  character(8) :: c
  integer :: i
  !ERROR: A substring cannot appear in a SHARED clause
  !$omp parallel shared(c(2:3))
  !$omp end parallel
  !ERROR: A substring cannot appear in a PRIVATE clause
  !$omp parallel private(c(2:3))
  !$omp end parallel
  !ERROR: A substring cannot appear in a FIRSTPRIVATE clause
  !$omp parallel firstprivate(c(2:3))
  !$omp end parallel
  !ERROR: A substring cannot appear in a LASTPRIVATE clause
  !$omp parallel do lastprivate(c(2:3))
  do i = 1, 2
  end do
  !$omp end parallel do
end subroutine

subroutine mapping(c, a)
  character(8) :: c, a(2)
  !$omp target map(tofrom: c(2:3), a(1)(2:3))
  !$omp end target
  !$omp target update to(c(2:3))
  !$omp target update from(a(1)(2:3))
end subroutine
