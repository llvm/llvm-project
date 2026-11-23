! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp

subroutine foo(x)
  integer :: x
  !ERROR: One of FROM or TO clauses is required on TARGET UPDATE directive
  !$omp target update

  !ERROR: One of FROM or TO clauses is required on TARGET UPDATE directive
  !$omp target update nowait

  !$omp target update to(x) nowait

  !ERROR: At most one NOWAIT clause can appear on TARGET UPDATE directive
  !$omp target update to(x) nowait nowait

  !ERROR: A list item ('x') can only appear in a TO or FROM clause, but not in both.
  !BECAUSE: 'x' appears in the TO clause.
  !BECAUSE: 'x' appears in the FROM clause.
  !$omp target update to(x) from(x)

end subroutine
