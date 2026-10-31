! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags -fopenmp-version=51
! Semantic checks for incompatible memory orderings on atomic operations.

subroutine test_atomic_read_mem_order()
  integer :: x, v

  ! Valid orderings for read
  !$omp atomic read seq_cst
  v = x
  !$omp atomic read acquire
  v = x
  !$omp atomic read relaxed
  v = x

  ! Invalid orderings for read
  !WARNING: An ATOMIC READ operation must not have RELEASE as the memory order, using RELAXED
  !$omp atomic read release
  v = x
  !$omp atomic read acq_rel
  v = x
end subroutine

subroutine test_atomic_write_mem_order()
  integer :: x, v

  ! Valid orderings for write
  !$omp atomic write seq_cst
  x = v
  !$omp atomic write release
  x = v
  !$omp atomic write relaxed
  x = v

  ! Invalid orderings for write
  !WARNING: An ATOMIC WRITE operation must not have ACQUIRE as the memory order, using RELAXED
  !$omp atomic write acquire
  x = v
  !$omp atomic write acq_rel
  x = v
end subroutine

subroutine test_atomic_update_mem_order()
  integer :: x

  ! Valid orderings for update
  !$omp atomic update seq_cst
  x = x + 1
  !$omp atomic update release
  x = x + 1
  !$omp atomic update relaxed
  x = x + 1

  ! No restrictions on update in OpenMP 5.1
  !$omp atomic update acquire
  x = x + 1
  !$omp atomic update acq_rel
  x = x + 1
end subroutine

subroutine test_atomic_capture_mem_order()
  integer :: x, v

  ! Valid orderings for capture (all are allowed, but some produce warnings)
  !$omp atomic capture seq_cst
  v = x
  x = x + 1
  !$omp end atomic
  !$omp atomic capture acquire
  v = x
  x = x + 1
  !$omp end atomic
  !$omp atomic capture relaxed
  v = x
  x = x + 1
  !$omp end atomic

  ! Capture with release/acq_rel: valid, no restrictions
  !$omp atomic capture release
  v = x
  x = x + 1
  !$omp end atomic
  !$omp atomic capture acq_rel
  v = x
  x = x + 1
  !$omp end atomic

  ! Trigger an error so test_errors.py captures stderr
  !ERROR: ATOMIC READ cannot have COMPARE or CAPTURE clauses
  !$omp atomic read capture
  v = x
  x = x + 1
  !$omp end atomic
end subroutine
