! REQUIRES: openmp_runtime
! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags -fopenmp-version=51
  use omp_lib
  implicit none
  ! Check atomic compare. This combines elements from multiple other "atomic*.f90", as
  ! to avoid having several files with just a few lines in them. atomic compare needs
  ! higher openmp version than the others, so need a separate file.
  

  real a, b, c
  a = 1.0
  b = 2.0
  c = 3.0
  !$omp parallel num_threads(4)
  ! First a few things that should compile without error.
  !$omp atomic seq_cst, compare
  if (b .eq. a) then
     b = c
  end if

  !$omp atomic seq_cst compare
  if (a .eq. b) a = c
  !$omp end atomic

  !$omp atomic compare acquire hint(OMP_LOCK_HINT_CONTENDED)
  if (b .eq. a) b = c

  !$omp atomic release hint(OMP_LOCK_HINT_UNCONTENDED) compare
  if (b .eq. a) b = c

  !$omp atomic compare seq_cst
  if (b .eq. c) b = a

  !$omp atomic hint(1) acq_rel compare
  if (b .eq. a) b = c
  !$omp end atomic

  !$omp atomic hint(1) acq_rel compare fail(release)
  if (c .eq. a) a = b
  !$omp end atomic

  !$omp atomic compare fail(release)
  if (c .eq. a) a = b
  !$omp end atomic

  ! Check for error conditions:
  !ERROR: More than one memory order clause not allowed on OpenMP ATOMIC construct
  !ERROR: At most one SEQ_CST clause can appear on the COMPARE directive
  !$omp atomic seq_cst seq_cst compare
  if (b .eq. c) b = a
  !ERROR: More than one memory order clause not allowed on OpenMP ATOMIC construct
  !ERROR: At most one SEQ_CST clause can appear on the COMPARE directive
  !$omp atomic compare seq_cst seq_cst
  if (b .eq. c) b = a
  !ERROR: More than one memory order clause not allowed on OpenMP ATOMIC construct
  !ERROR: At most one SEQ_CST clause can appear on the COMPARE directive
  !$omp atomic seq_cst compare seq_cst
  if (b .eq. c) b = a

  !ERROR: More than one memory order clause not allowed on OpenMP ATOMIC construct
  !ERROR: At most one ACQUIRE clause can appear on the COMPARE directive
  !$omp atomic acquire acquire compare
  if (b .eq. c) b = a
  !ERROR: More than one memory order clause not allowed on OpenMP ATOMIC construct
  !ERROR: At most one ACQUIRE clause can appear on the COMPARE directive
  !$omp atomic compare acquire acquire
  if (b .eq. c) b = a
  !ERROR: More than one memory order clause not allowed on OpenMP ATOMIC construct
  !ERROR: At most one ACQUIRE clause can appear on the COMPARE directive
  !$omp atomic acquire compare acquire
  if (b .eq. c) b = a

  !ERROR: More than one memory order clause not allowed on OpenMP ATOMIC construct
  !ERROR: At most one RELAXED clause can appear on the COMPARE directive
  !$omp atomic relaxed relaxed compare
  if (b .eq. c) b = a
  !ERROR: More than one memory order clause not allowed on OpenMP ATOMIC construct
  !ERROR: At most one RELAXED clause can appear on the COMPARE directive
  !$omp atomic compare relaxed relaxed
  if (b .eq. c) b = a
  !ERROR: More than one memory order clause not allowed on OpenMP ATOMIC construct
  !ERROR: At most one RELAXED clause can appear on the COMPARE directive
  !$omp atomic relaxed compare relaxed
  if (b .eq. c) b = a

  !ERROR: More than one FAIL clause not allowed on OpenMP ATOMIC construct
  !$omp atomic fail(release) compare fail(release)
  if (c .eq. a) a = b
  !$omp end atomic

  !$omp end parallel
end
