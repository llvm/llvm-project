! RUN: %python %S/test_errors.py %s %flang -fopenmp

! Test the effect to name resolution from illegal clause

  !a = 1.0
  b = 2
  !$omp parallel private(a) shared(b)
  a = 3.
  b = 4
  !ERROR: LASTPRIVATE clause is not allowed on the PARALLEL directive
  !ERROR: 'a' appears in more than one data-sharing clause on the same OpenMP directive
  !$omp parallel private(a) shared(b) lastprivate(a)
  a = 5.
  b = 6
  !$omp end parallel
  !$omp end parallel
  print *,a, b
end
