! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! Semantic checks for OpenMP 5.0 standard 2.17.7 atomic Construct.

use omp_lib
  implicit none
  integer :: i, j = 10, k=-100, a
! 2.17.7.1
! Handled inside parser.
! OpenMP constructs may not be encountered during execution of an atomic region

! 2.17.7.2
! At most one memory-order-clause may appear on the construct.

!READ
  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one SEQ_CST clause can appear on the READ directive
  !$omp atomic seq_cst seq_cst read
    i = j
  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one SEQ_CST clause can appear on the READ directive
  !$omp atomic read seq_cst seq_cst
    i = j
  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one SEQ_CST clause can appear on the READ directive
  !$omp atomic seq_cst read seq_cst
    i = j

  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one ACQUIRE clause can appear on the READ directive
  !$omp atomic acquire acquire read
    i = j
  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one ACQUIRE clause can appear on the READ directive
  !$omp atomic read acquire acquire
    i = j
  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one ACQUIRE clause can appear on the READ directive
  !$omp atomic acquire read acquire
    i = j

  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one RELAXED clause can appear on the READ directive
  !$omp atomic relaxed relaxed read
    i = j
  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one RELAXED clause can appear on the READ directive
  !$omp atomic read relaxed relaxed
    i = j
  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one RELAXED clause can appear on the READ directive
  !$omp atomic relaxed read relaxed
    i = j

!UPDATE
  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one SEQ_CST clause can appear on the UPDATE directive
  !$omp atomic seq_cst seq_cst update
    i = j
  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one SEQ_CST clause can appear on the UPDATE directive
  !$omp atomic update seq_cst seq_cst
    i = j
  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one SEQ_CST clause can appear on the UPDATE directive
  !$omp atomic seq_cst update seq_cst
    i = j

  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one RELEASE clause can appear on the UPDATE directive
  !$omp atomic release release update
    i = j
  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one RELEASE clause can appear on the UPDATE directive
  !$omp atomic update release release
    i = j
  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one RELEASE clause can appear on the UPDATE directive
  !$omp atomic release update release
    i = j

  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one RELAXED clause can appear on the UPDATE directive
  !$omp atomic relaxed relaxed update
    i = j
  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one RELAXED clause can appear on the UPDATE directive
  !$omp atomic update relaxed relaxed
    i = j
  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one RELAXED clause can appear on the UPDATE directive
  !$omp atomic relaxed update relaxed
    i = j

!CAPTURE
  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one SEQ_CST clause can appear on the CAPTURE directive
  !$omp atomic seq_cst seq_cst capture
    i = j
    j = k
  !$omp end atomic
  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one SEQ_CST clause can appear on the CAPTURE directive
  !$omp atomic capture seq_cst seq_cst
    i = j
    j = k
  !$omp end atomic

  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one SEQ_CST clause can appear on the CAPTURE directive
  !$omp atomic seq_cst capture seq_cst
    i = j
    j = k
  !$omp end atomic

  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one RELEASE clause can appear on the CAPTURE directive
  !$omp atomic release release capture
    i = j
    j = k
  !$omp end atomic

  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one RELEASE clause can appear on the CAPTURE directive
  !$omp atomic capture release release
    i = j
    j = k
  !$omp end atomic

  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one RELEASE clause can appear on the CAPTURE directive
  !$omp atomic release capture release
    i = j
    j = k
  !$omp end atomic

  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one RELAXED clause can appear on the CAPTURE directive
  !$omp atomic relaxed relaxed capture
    i = j
    j = k
  !$omp end atomic

  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one RELAXED clause can appear on the CAPTURE directive
  !$omp atomic capture relaxed relaxed
    i = j
    j = k
  !$omp end atomic

  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one RELAXED clause can appear on the CAPTURE directive
  !$omp atomic relaxed capture relaxed
    i = j
    j = k
  !$omp end atomic

  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one ACQ_REL clause can appear on the CAPTURE directive
  !$omp atomic acq_rel acq_rel capture
    i = j
    j = k
  !$omp end atomic

  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one ACQ_REL clause can appear on the CAPTURE directive
  !$omp atomic capture acq_rel acq_rel
    i = j
    j = k
  !$omp end atomic

  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one ACQ_REL clause can appear on the CAPTURE directive
  !$omp atomic acq_rel capture acq_rel
    i = j
    j = k
  !$omp end atomic

  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one ACQUIRE clause can appear on the CAPTURE directive
  !$omp atomic acquire acquire capture
    i = j
    j = k
  !$omp end atomic

  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one ACQUIRE clause can appear on the CAPTURE directive
  !$omp atomic capture acquire acquire
    i = j
    j = k
  !$omp end atomic

  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one ACQUIRE clause can appear on the CAPTURE directive
  !$omp atomic acquire capture acquire
    i = j
    j = k
  !$omp end atomic

!WRITE
  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one SEQ_CST clause can appear on the WRITE directive
  !$omp atomic seq_cst seq_cst write
    i = j
  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one SEQ_CST clause can appear on the WRITE directive
  !$omp atomic write seq_cst seq_cst
    i = j

  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one SEQ_CST clause can appear on the WRITE directive
  !$omp atomic seq_cst write seq_cst
    i = j

  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one RELEASE clause can appear on the WRITE directive
  !$omp atomic release release write
    i = j
  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one RELEASE clause can appear on the WRITE directive
  !$omp atomic write release release
    i = j
  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one RELEASE clause can appear on the WRITE directive
  !$omp atomic release write release
    i = j
  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one RELAXED clause can appear on the WRITE directive
  !$omp atomic relaxed relaxed write
    i = j
  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one RELAXED clause can appear on the WRITE directive
  !$omp atomic write relaxed relaxed
    i = j
  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one RELAXED clause can appear on the WRITE directive
  !$omp atomic relaxed write relaxed
    i = j

!No atomic-clause
  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one RELAXED clause can appear on the ATOMIC directive
  !$omp atomic relaxed relaxed
    i = j
  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one SEQ_CST clause can appear on the ATOMIC directive
  !$omp atomic seq_cst seq_cst
    i = j
  !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
  !ERROR: At most one RELEASE clause can appear on the ATOMIC directive
  !$omp atomic release release
    i = j

! 2.17.7.3
! At most one hint clause may appear on the construct.

  !ERROR: At most one HINT clause can appear on the READ directive
  !$omp atomic hint(omp_sync_hint_speculative) hint(omp_sync_hint_speculative) read
    i = j
  !ERROR: At most one HINT clause can appear on the READ directive
  !$omp atomic hint(omp_sync_hint_nonspeculative) read hint(omp_sync_hint_nonspeculative)
    i = j
  !ERROR: At most one HINT clause can appear on the READ directive
  !$omp atomic read hint(omp_sync_hint_uncontended) hint (omp_sync_hint_uncontended)
    i = j
  !ERROR: At most one HINT clause can appear on the WRITE directive
  !$omp atomic hint(omp_sync_hint_contended) hint(omp_sync_hint_speculative) write
    i = j
  !ERROR: At most one HINT clause can appear on the WRITE directive
  !$omp atomic hint(omp_sync_hint_nonspeculative) write hint(omp_sync_hint_nonspeculative)
    i = j
  !ERROR: At most one HINT clause can appear on the WRITE directive
  !$omp atomic write hint(omp_sync_hint_none) hint (omp_sync_hint_uncontended)
    i = j
  !ERROR: At most one HINT clause can appear on the WRITE directive
  !$omp atomic hint(omp_sync_hint_contended) hint(omp_sync_hint_speculative) write
    i = j
  !ERROR: At most one HINT clause can appear on the WRITE directive
  !$omp atomic hint(omp_sync_hint_nonspeculative) write hint(omp_sync_hint_nonspeculative)
    i = j
  !ERROR: At most one HINT clause can appear on the WRITE directive
  !$omp atomic write hint(omp_sync_hint_none) hint (omp_sync_hint_uncontended)
    i = j
  !ERROR: At most one HINT clause can appear on the UPDATE directive
  !$omp atomic hint(omp_sync_hint_contended) hint(omp_sync_hint_speculative) update
    i = j
  !ERROR: At most one HINT clause can appear on the UPDATE directive
  !$omp atomic hint(omp_sync_hint_nonspeculative) update hint(omp_sync_hint_nonspeculative)
    i = j
  !ERROR: At most one HINT clause can appear on the UPDATE directive
  !$omp atomic update hint(omp_sync_hint_none) hint (omp_sync_hint_uncontended)
    i = j
  !ERROR: At most one HINT clause can appear on the ATOMIC directive
  !$omp atomic hint(omp_sync_hint_contended) hint(omp_sync_hint_speculative)
    i = j
  !ERROR: At most one HINT clause can appear on the ATOMIC directive
  !$omp atomic hint(omp_sync_hint_none) hint(omp_sync_hint_nonspeculative)
    i = j
  !ERROR: At most one HINT clause can appear on the ATOMIC directive
  !$omp atomic hint(omp_sync_hint_none) hint (omp_sync_hint_uncontended)
    i = j

  !ERROR: At most one HINT clause can appear on the CAPTURE directive
  !$omp atomic hint(omp_sync_hint_contended) hint(omp_sync_hint_speculative) capture
    i = j
    j = k
  !$omp end atomic
  !ERROR: At most one HINT clause can appear on the CAPTURE directive
  !$omp atomic hint(omp_sync_hint_nonspeculative) capture hint(omp_sync_hint_nonspeculative)
    i = j
    j = k
  !$omp end atomic
  !ERROR: At most one HINT clause can appear on the CAPTURE directive
  !$omp atomic capture hint(omp_sync_hint_none) hint (omp_sync_hint_uncontended)
    i = j
    j = k
  !$omp end atomic
! 2.17.7.4
! If atomic-clause is read then memory-order-clause must not be acq_rel or release.

  !ERROR: Clause ACQ_REL is not allowed if clause READ appears on the ATOMIC directive
  !$omp atomic acq_rel read
    i = j
  !ERROR: Clause ACQ_REL is not allowed if clause READ appears on the ATOMIC directive
  !$omp atomic read acq_rel
    i = j

  !ERROR: Clause RELEASE is not allowed if clause READ appears on the ATOMIC directive
  !$omp atomic release read
    i = j
  !ERROR: Clause RELEASE is not allowed if clause READ appears on the ATOMIC directive
  !$omp atomic read release
    i = j

! 2.17.7.5
! If atomic-clause is write then memory-order-clause must not be acq_rel or acquire.

  !ERROR: Clause ACQ_REL is not allowed if clause WRITE appears on the ATOMIC directive
  !$omp atomic acq_rel write
    i = j
  !ERROR: Clause ACQ_REL is not allowed if clause WRITE appears on the ATOMIC directive
  !$omp atomic write acq_rel
    i = j

  !ERROR: Clause ACQUIRE is not allowed if clause WRITE appears on the ATOMIC directive
  !$omp atomic acquire write
    i = j
  !ERROR: Clause ACQUIRE is not allowed if clause WRITE appears on the ATOMIC directive
  !$omp atomic write acquire
    i = j


! 2.17.7.6
! If atomic-clause is update or not present then memory-order-clause must not be acq_rel or acquire.

  !ERROR: Clause ACQ_REL is not allowed if clause UPDATE appears on the ATOMIC directive
  !$omp atomic acq_rel update
    i = j
  !ERROR: Clause ACQ_REL is not allowed if clause UPDATE appears on the ATOMIC directive
  !$omp atomic update acq_rel
    i = j

  !ERROR: Clause ACQUIRE is not allowed if clause UPDATE appears on the ATOMIC directive
  !$omp atomic acquire update
    i = j

  !ERROR: Clause ACQUIRE is not allowed if clause UPDATE appears on the ATOMIC directive
  !$omp atomic update acquire
    i = j

  !ERROR: Clause ACQ_REL is not allowed on the ATOMIC directive
  !$omp atomic acq_rel
    i = j

  !ERROR: Clause ACQUIRE is not allowed on the ATOMIC directive
  !$omp atomic acquire
    i = j
end program

