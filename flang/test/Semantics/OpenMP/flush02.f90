! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags -fopenmp-version=51

! Check OpenMP 5.0 - 2.17.8 flush Construct
! Restriction -
! If memory-order-clause is release, acquire, or acq_rel, list items must not be specified on the flush directive.

use omp_lib
  implicit none

  TYPE someStruct
    REAL :: rr
  end TYPE
  integer :: i, a, b
  real, DIMENSION(10) :: array
  TYPE(someStruct) :: structObj

  a = 1.0
  !$omp parallel num_threads(4)
  !No list flushes all.
  if (omp_get_thread_num() == 1) THEN
    !$omp flush
  END IF

  array = (/1, 2, 3, 4, 5, 6, 7, 8, 9, 10/)
  !Only memory-order-clauses.
  if (omp_get_thread_num() == 1) THEN
    ! Not allowed clauses.
    !$omp flush seq_cst
    !ERROR: RELAXED clause is not allowed on the FLUSH directive
    !$omp flush relaxed

    ! Not allowed more than once.
    !ERROR: At most one ACQ_REL clause can appear on the FLUSH directive
    !$omp flush acq_rel acq_rel
    !ERROR: At most one RELEASE clause can appear on the FLUSH directive
    !$omp flush release release
    !ERROR: At most one ACQUIRE clause can appear on the FLUSH directive
    !$omp flush acquire acquire

    ! Mix of allowed and not allowed.
    !$omp flush seq_cst acquire
  END IF

  array = (/1, 2, 3, 4, 5, 6, 7, 8, 9, 10/)
  ! No memory-order-clause only list-items.
  if (omp_get_thread_num() == 2) THEN
    !$omp flush (a)
    !$omp flush (i, a, b)
    !$omp flush (array, structObj%rr)
    ! Too many flush with repeating list items.
    !$omp flush (i, a, b, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, b, b, b, b)
    !ERROR: No explicit type declared for 'notpresentitem'
    !$omp flush (notPresentItem)
  END IF

  array = (/1, 2, 3, 4, 5, 6, 7, 8, 9, 10/)
  if (omp_get_thread_num() == 3) THEN
    !ERROR: If memory-order-clause is RELEASE, ACQUIRE, or ACQ_REL, list items must not be specified on the FLUSH directive
    !$omp flush acq_rel (array)
    !ERROR: If memory-order-clause is RELEASE, ACQUIRE, or ACQ_REL, list items must not be specified on the FLUSH directive
    !$omp flush acq_rel (array, a, i)

    array = (/1, 2, 3, 4, 5, 6, 7, 8, 9, 10/)
    !ERROR: If memory-order-clause is RELEASE, ACQUIRE, or ACQ_REL, list items must not be specified on the FLUSH directive
    !$omp flush release (array)
    !ERROR: If memory-order-clause is RELEASE, ACQUIRE, or ACQ_REL, list items must not be specified on the FLUSH directive
    !$omp flush release (array, a)

    array = (/1, 2, 3, 4, 5, 6, 7, 8, 9, 10/)
    !ERROR: If memory-order-clause is RELEASE, ACQUIRE, or ACQ_REL, list items must not be specified on the FLUSH directive
    !$omp flush acquire (array)
    !ERROR: If memory-order-clause is RELEASE, ACQUIRE, or ACQ_REL, list items must not be specified on the FLUSH directive
    !$omp flush acquire (array, a, structObj%rr)
  END IF
  !$omp end parallel

  !$omp parallel num_threads(4)
    array = (/1, 2, 3, 4, 5, 6, 7, 8, 9, 10/)
    !WARNING: OpenMP directive MASTER has been deprecated, please use MASKED instead.
    !$omp master
      !$omp flush (array)
    !$omp end master
  !$omp end parallel
end

