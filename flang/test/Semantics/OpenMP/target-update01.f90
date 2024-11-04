! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp

subroutine foo(x)
  integer :: x
  !ERROR: At least one motion-clause (TO/FROM) must be specified on TARGET UPDATE construct.
  !$omp target update

  !ERROR: At least one motion-clause (TO/FROM) must be specified on TARGET UPDATE construct.
  !$omp target update nowait

  !$omp target update to(x) nowait

  !ERROR: At most one NOWAIT clause can appear on the TARGET UPDATE directive
  !$omp target update to(x) nowait nowait

  !ERROR: A list item ('x') can only appear in a TO or FROM clause, but not in both.
  !BECAUSE: 'x' appears in the TO clause.
  !BECAUSE: 'x' appears in the FROM clause.
  !$omp target update to(x) from(x)

end subroutine
