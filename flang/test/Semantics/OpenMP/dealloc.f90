! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp

! Test to check that no errors are present when allocate statements
! are applied on privatised variables.

subroutine s
  implicit none
  double precision,allocatable,dimension(:) :: r
  !$omp parallel private(r)
  allocate(r(1))
  deallocate(r)
  !$omp end parallel
end subroutine
