! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=50
! OpenMP Version 5.0
! 2.4 Requires directive
! The 'lexically before any device construct' restriction is scoped to a
! program unit: a device construct (here a target teams distribute parallel do
! loop) in one program unit must not make a REQUIRES directive in a separate
! program unit ill-formed.

subroutine f
  !$omp target teams distribute parallel do
  do i=1, 10
  end do
  !$omp end target teams distribute parallel do
end subroutine f

subroutine g
  !$omp requires unified_shared_memory
  !$omp requires unified_address
end subroutine g