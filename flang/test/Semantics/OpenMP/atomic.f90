! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang -fopenmp %openmp_flags
use omp_lib
! Check OpenMP 2.13.6 atomic Construct

  a = 1.0
  !$omp parallel num_threads(4)
  !$omp atomic seq_cst, read
  b = a

  !$omp atomic seq_cst write
  a = b
  !$omp end atomic

  !ERROR: ACQUIRE clause is not allowed on directive ATOMIC in OpenMP v3.1, try -fopenmp-version=50
  !ERROR: HINT clause is not allowed on directive ATOMIC in OpenMP v3.1, try -fopenmp-version=50
  !$omp atomic read acquire hint(OMP_LOCK_HINT_CONTENDED)
  a = b

  !ERROR: RELEASE clause is not allowed on directive ATOMIC in OpenMP v3.1, try -fopenmp-version=50
  !ERROR: HINT clause is not allowed on directive ATOMIC in OpenMP v3.1, try -fopenmp-version=50
  !$omp atomic release hint(OMP_LOCK_HINT_UNCONTENDED) write
  a = b

  !$omp atomic capture seq_cst
  b = a
  a = a + 1
  !$omp end atomic

  !ERROR: HINT clause is not allowed on directive ATOMIC in OpenMP v3.1, try -fopenmp-version=50
  !ERROR: ACQ_REL clause is not allowed on directive ATOMIC in OpenMP v3.1, try -fopenmp-version=50
  !$omp atomic hint(1) acq_rel capture
  b = a
  a = a + 1
  !$omp end atomic

  !ERROR: At most one clause from the 'atomic' group is allowed on ATOMIC construct
  !$omp atomic read write
  !ERROR: Atomic expression a+1._4 should be a variable
  a = a + 1

  !$omp atomic
  a = a + 1
  !ERROR: NUM_THREADS clause is not allowed on the ATOMIC directive
  !$omp atomic num_threads(4)
  a = a + 1

  !ERROR: ATOMIC UPDATE operation with CAPTURE should contain two statements
  !ERROR: NUM_THREADS clause is not allowed on the ATOMIC directive
  !$omp atomic capture num_threads(4)
  a = a + 1

  !ERROR: RELAXED clause is not allowed on directive ATOMIC in OpenMP v3.1, try -fopenmp-version=50
  !$omp atomic relaxed
  a = a + 1

  !$omp end parallel
end
