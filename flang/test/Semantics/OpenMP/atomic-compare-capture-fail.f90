! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=52

! Atomic compare-capture with each of the 5 effective memory orders combined
! with each of the 3 valid FAIL arguments (SEQ_CST, ACQUIRE, RELAXED): all 15
! combinations are valid, so no diagnostics are expected.

!===----------------------------------------------------------------------===!
! Success order: SEQ_CST
!===----------------------------------------------------------------------===!

subroutine seq_cst_fail_seq_cst(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic update compare capture seq_cst fail(seq_cst)
  v = x
  if (x == e) x = d
  !$omp end atomic
end subroutine

subroutine seq_cst_fail_acquire(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic update compare capture seq_cst fail(acquire)
  if (x == e) then
    x = d
  else
    v = x
  end if
  !$omp end atomic
end subroutine

subroutine seq_cst_fail_relaxed(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic update compare capture seq_cst fail(relaxed)
  v = x
  if (x == e) x = d
  !$omp end atomic
end subroutine

!===----------------------------------------------------------------------===!
! Success order: ACQ_REL
!===----------------------------------------------------------------------===!

subroutine acq_rel_fail_seq_cst(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic update compare capture acq_rel fail(seq_cst)
  v = x
  if (x == e) x = d
  !$omp end atomic
end subroutine

subroutine acq_rel_fail_acquire(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic update compare capture acq_rel fail(acquire)
  v = x
  if (x == e) x = d
  !$omp end atomic
end subroutine

subroutine acq_rel_fail_relaxed(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic update compare capture acq_rel fail(relaxed)
  v = x
  if (x == e) x = d
  !$omp end atomic
end subroutine

!===----------------------------------------------------------------------===!
! Success order: RELEASE
!===----------------------------------------------------------------------===!

subroutine release_fail_seq_cst(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic update compare capture release fail(seq_cst)
  v = x
  if (x == e) x = d
  !$omp end atomic
end subroutine

subroutine release_fail_acquire(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic update compare capture release fail(acquire)
  v = x
  if (x == e) x = d
  !$omp end atomic
end subroutine

subroutine release_fail_relaxed(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic update compare capture release fail(relaxed)
  v = x
  if (x == e) x = d
  !$omp end atomic
end subroutine

!===----------------------------------------------------------------------===!
! Success order: ACQUIRE
!===----------------------------------------------------------------------===!

subroutine acquire_fail_seq_cst(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic update compare capture acquire fail(seq_cst)
  v = x
  if (x == e) x = d
  !$omp end atomic
end subroutine

subroutine acquire_fail_acquire(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic update compare capture acquire fail(acquire)
  v = x
  if (x == e) x = d
  !$omp end atomic
end subroutine

subroutine acquire_fail_relaxed(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic update compare capture acquire fail(relaxed)
  v = x
  if (x == e) x = d
  !$omp end atomic
end subroutine

!===----------------------------------------------------------------------===!
! Success order: RELAXED
!===----------------------------------------------------------------------===!

subroutine relaxed_fail_seq_cst(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic update compare capture relaxed fail(seq_cst)
  v = x
  if (x == e) x = d
  !$omp end atomic
end subroutine

subroutine relaxed_fail_acquire(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic update compare capture relaxed fail(acquire)
  v = x
  if (x == e) x = d
  !$omp end atomic
end subroutine

subroutine relaxed_fail_relaxed(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic update compare capture relaxed fail(relaxed)
  v = x
  if (x == e) x = d
  !$omp end atomic
end subroutine
