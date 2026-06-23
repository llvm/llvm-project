! REQUIRES: openmp_runtime
! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags -fopenmp-version=51
  use omp_lib
  implicit none
  ! Check atomic compare. This combines elements from multiple other "atomic*.f90", as
  ! to avoid having several files with just a few lines in them. atomic compare needs
  ! higher openmp version than the others, so need a separate file.


  real a, b, c
  logical :: r, s
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

  ! Less-than comparison.
  !$omp atomic compare
  if (b .lt. a) b = c

  ! Greater-than comparison.
  !$omp atomic compare
  if (b .gt. a) b = c

  ! Two-statement form: r = cond; if (r) update.
  !$omp atomic compare
  r = b .eq. a
  if (r) b = c
  !$omp end atomic

  ! Check for error conditions:
  !ERROR: At most one SEQ_CST clause can appear on the ATOMIC directive
  !$omp atomic seq_cst seq_cst compare
  if (b .eq. c) b = a
  !ERROR: At most one SEQ_CST clause can appear on the ATOMIC directive
  !$omp atomic compare seq_cst seq_cst
  if (b .eq. c) b = a
  !ERROR: At most one SEQ_CST clause can appear on the ATOMIC directive
  !$omp atomic seq_cst compare seq_cst
  if (b .eq. c) b = a

  !ERROR: At most one ACQUIRE clause can appear on the ATOMIC directive
  !$omp atomic acquire acquire compare
  if (b .eq. c) b = a
  !ERROR: At most one ACQUIRE clause can appear on the ATOMIC directive
  !$omp atomic compare acquire acquire
  if (b .eq. c) b = a
  !ERROR: At most one ACQUIRE clause can appear on the ATOMIC directive
  !$omp atomic acquire compare acquire
  if (b .eq. c) b = a

  !ERROR: At most one RELAXED clause can appear on the ATOMIC directive
  !$omp atomic relaxed relaxed compare
  if (b .eq. c) b = a
  !ERROR: At most one RELAXED clause can appear on the ATOMIC directive
  !$omp atomic compare relaxed relaxed
  if (b .eq. c) b = a
  !ERROR: At most one RELAXED clause can appear on the ATOMIC directive
  !$omp atomic relaxed compare relaxed
  if (b .eq. c) b = a

  !ERROR: At most one FAIL clause can appear on the ATOMIC directive
  !$omp atomic fail(release) compare fail(release)
  if (c .eq. a) a = b
  !$omp end atomic

  ! The /= operator is not valid for atomic compare.
  !$omp atomic compare
  !ERROR: The /= operator is not a valid condition for ATOMIC operation
  if (b .ne. a) b = c

  ! The <= operator is not valid for atomic compare.
  !$omp atomic compare
  !ERROR: The <= operator is not a valid condition for ATOMIC operation
  if (b .le. a) b = c

  ! The >= operator is not valid for atomic compare.
  !$omp atomic compare
  !ERROR: The >= operator is not a valid condition for ATOMIC operation
  if (b .ge. a) b = c

  ! ELSE branch is not allowed.
  !$omp atomic compare
  if (b .eq. a) then
    b = c
  else
  !ERROR: In ATOMIC UPDATE COMPARE the update statement should not have an ELSE branch
    a = b
  end if

  ! Not a conditional statement.
  !ERROR: In ATOMIC UPDATE COMPARE the update statement should be a conditional statement
  !$omp atomic compare
  b = c

  ! Too many statements.
  !ERROR: ATOMIC UPDATE COMPARE operation should contain one or two statements
  !$omp atomic compare
  r = b .eq. a
  if (r) b = c
  a = b
  !$omp end atomic

  ! Two-statement form with wrong condition variable.
  !$omp atomic compare
  r = b .eq. a
  !ERROR: In ATOMIC UPDATE COMPARE the conditional statement must use r as the condition
  if (s) b = c
  !$omp end atomic

  ! Neither argument of the condition is the target of the assignment.
  !$omp atomic compare
  !ERROR: An argument of the == operator should be the target of the assignment
  if (a .eq. c) b = c

  ! First statement is not a comparison, condition uses wrong variable.
  !$omp atomic compare
  b = c
  !ERROR: In ATOMIC UPDATE COMPARE the conditional statement must use b as the condition
  if (r) b = c
  !$omp end atomic

  !$omp end parallel
end
