! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=50
! OpenMP Version 5.0
! 2.4 Requires directive
! The 'lexically before any device construct' restriction is scoped to a
! program unit: a device construct (here a target parallel region) in one
! program unit must not make a REQUIRES directive in a separate program unit
! ill-formed.

subroutine f
  !$omp target parallel
  !$omp end target parallel
end subroutine f

subroutine g
  !$omp requires unified_shared_memory
  !$omp requires unified_address
end subroutine g