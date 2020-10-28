! RUN: %S/test_errors.sh %s %t %f18 -fopenmp
use omp_lib
! Check OpenMP 2.13.6 atomic Construct

  a = 1.0
  !$omp parallel num_threads(4)
  !$omp atomic seq_cst, read
  b = a

  !$omp atomic seq_cst write
  a = b
  !$omp end atomic

  !$omp atomic read acquire hint(OMP_LOCK_HINT_CONTENDED)
  a = b

  !$omp atomic release hint(OMP_LOCK_HINT_UNCONTENDED) write
  a = b

  !$omp atomic capture seq_cst
  b = a
  a = a + 1
  !$omp end atomic

  !$omp atomic hint(1) acq_rel capture
  b = a
  a = a + 1
  !$omp end atomic

  !ERROR: expected end of line
  !ERROR: expected end of line
  !$omp atomic read write
  a = a + 1

  !$omp atomic
  a = a + 1

  !$omp atomic relaxed
  a = a + 1

  !$omp end parallel
end
