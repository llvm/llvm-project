! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51
  use omp_lib
  implicit none
  ! Check atomic compare. This combines elements from multiple other "atomic*.f90", as
  ! to avoid having several files with just a few lines in them. atomic compare needs
  ! higher openmp version than the others, so need a separate file.
  

  real a, b
  logical r
  a = 1.0
  b = 2.0
  !$omp parallel num_threads(4)
  ! First a few things that should compile without error.
  !$omp atomic seq_cst, compare
  r = b .ne. a

  !$omp atomic seq_cst compare
  r = a .ge. b
  !$omp end atomic

  !$omp atomic compare acquire hint(OMP_LOCK_HINT_CONTENDED)
  r = a .lt. b

  !$omp atomic release hint(OMP_LOCK_HINT_UNCONTENDED) compare
  r = a .gt. b

  !$omp atomic compare seq_cst
  r = b .ne. a

  !$omp atomic hint(1) acq_rel compare
  r = b .eq. a
  !$omp end atomic

  ! Check for error conidtions:
  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one SEQ_CST clause can appear on the COMPARE directive
  !$omp atomic seq_cst seq_cst compare
  r = a .le. b
  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one SEQ_CST clause can appear on the COMPARE directive
  !$omp atomic compare seq_cst seq_cst
  r = b .gt. a
  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one SEQ_CST clause can appear on the COMPARE directive
  !$omp atomic seq_cst compare seq_cst
  r = b .ge. b

  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one ACQUIRE clause can appear on the COMPARE directive
  !$omp atomic acquire acquire compare
  r = a .le. b
  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one ACQUIRE clause can appear on the COMPARE directive
  !$omp atomic compare acquire acquire
  r = b .gt. a
  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one ACQUIRE clause can appear on the COMPARE directive
  !$omp atomic acquire compare acquire
  r = b .ge. b

  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one RELAXED clause can appear on the COMPARE directive
  !$omp atomic relaxed relaxed compare
  r = a .le. b
  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one RELAXED clause can appear on the COMPARE directive
  !$omp atomic compare relaxed relaxed
  r = b .gt. a
  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one RELAXED clause can appear on the COMPARE directive
  !$omp atomic relaxed compare relaxed
  r = b .ge. b

  !$omp end parallel
end
