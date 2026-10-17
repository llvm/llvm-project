! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=52

! OpenMP 5.2 [15.8.3] extended-atomic Clauses: acq_rel and release cannot be
! specified as arguments to the fail clause, so the argument must be SEQ_CST,
! ACQUIRE, or RELAXED.

subroutine valid(x, e, d)
  integer :: x, e, d
  !$omp atomic compare fail(seq_cst)
  if (x == e) x = d
  !$omp atomic compare fail(acquire)
  if (x == e) x = d
  !$omp atomic compare fail(relaxed)
  if (x == e) x = d
  !$omp atomic seq_cst compare fail(relaxed)
  if (x == e) x = d
end

subroutine invalid(x, e, d)
  integer :: x, e, d
  !ERROR: The argument of the FAIL clause must be SEQ_CST, ACQUIRE, or RELAXED
  !$omp atomic compare fail(release)
  if (x == e) x = d
  !ERROR: The argument of the FAIL clause must be SEQ_CST, ACQUIRE, or RELAXED
  !$omp atomic compare fail(acq_rel)
  if (x == e) x = d
end
