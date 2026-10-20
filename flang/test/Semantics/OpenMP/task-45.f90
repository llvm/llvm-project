! REQUIRES: openmp_runtime
! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags -fopenmp-version=45

! Check that we still emit diagnostics for features from future versions
! that are allowed under a warning.

subroutine f
  use omp_lib, only: omp_event_handle_kind
  integer(kind=omp_event_handle_kind) :: e
  !ERROR: Clause MERGEABLE is not allowed if clause DETACH appears on the TASK directive
  !WARNING: DETACH clause is not allowed on TASK directive in OpenMP v4.5, try -fopenmp-version=50 [-Wopenmp-future]
  !$omp task mergeable detach(e)
  !$omp end task
end
