! RUN: %flang_fc1 -fopenmp -fopenmp-version=50 -fdebug-dump-parse-tree %s 2>&1 | FileCheck %s
! Ensure that requires atomic_default_mem_order is used to update atomic
! operations with no explicit memory order set. ACQ_REL clause tested here.
program requires
  implicit none
  !$omp requires atomic_default_mem_order(acq_rel)
  integer :: i, j

  ! ----------------------------------------------------------------------------
  ! READ
  ! ----------------------------------------------------------------------------

  ! CHECK-LABEL: OpenMPAtomicConstruct
  ! CHECK: OmpClause -> Read
  ! CHECK: OmpClause -> Acquire
  !$omp atomic read
  i = j

  ! CHECK-LABEL: OpenMPAtomicConstruct
  ! CHECK-NOT: OmpClause -> AcqRel
  ! CHECK: OmpClause -> Relaxed
  ! CHECK: OmpClause -> Read
  !$omp atomic relaxed read
  i = j

  ! CHECK-LABEL: OpenMPAtomicConstruct
  ! CHECK: OmpClause -> Read
  ! CHECK-NOT: OmpClause -> AcqRel
  ! CHECK: OmpClause -> Relaxed
  !$omp atomic read relaxed
  i = j
  
  ! ----------------------------------------------------------------------------
  ! WRITE
  ! ----------------------------------------------------------------------------

  ! CHECK-LABEL: OpenMPAtomicConstruct
  ! CHECK: OmpClause -> Write
  ! CHECK: OmpClause -> Release
  !$omp atomic write
  i = j

  ! CHECK-LABEL: OpenMPAtomicConstruct
  ! CHECK-NOT: OmpClause -> AcqRel
  ! CHECK: OmpClause -> Relaxed
  ! CHECK: OmpClause -> Write
  !$omp atomic relaxed write
  i = j

  ! CHECK-LABEL: OpenMPAtomicConstruct
  ! CHECK: OmpClause -> Write
  ! CHECK-NOT: OmpClause -> AcqRel
  ! CHECK: OmpClause -> Relaxed
  !$omp atomic write relaxed
  i = j

  ! ----------------------------------------------------------------------------
  ! UPDATE
  ! ----------------------------------------------------------------------------

  ! CHECK-LABEL: OpenMPAtomicConstruct
  ! CHECK: OmpClause -> Update
  ! CHECK: OmpClause -> Release
  !$omp atomic update
  i = i + j

  ! CHECK-LABEL: OpenMPAtomicConstruct
  ! CHECK-NOT: OmpClause -> AcqRel
  ! CHECK: OmpClause -> Relaxed
  ! CHECK: OmpClause -> Update
  !$omp atomic relaxed update
  i = i + j

  ! CHECK-LABEL: OpenMPAtomicConstruct
  ! CHECK: OmpClause -> Update
  ! CHECK-NOT: OmpClause -> AcqRel
  ! CHECK: OmpClause -> Relaxed
  !$omp atomic update relaxed
  i = i + j

  ! CHECK-LABEL: OpenMPAtomicConstruct
  ! CHECK: OmpClause -> Release
  !$omp atomic
  i = i + j

  ! CHECK-LABEL: OpenMPAtomicConstruct
  ! CHECK-NOT: OmpClause -> AcqRel
  ! CHECK: OmpClause -> Relaxed
  !$omp atomic relaxed
  i = i + j

  ! ----------------------------------------------------------------------------
  ! CAPTURE
  ! ----------------------------------------------------------------------------

  ! CHECK-LABEL: OpenMPAtomicConstruct
  ! CHECK: OmpClause -> Capture
  ! CHECK: OmpClause -> AcqRel
  !$omp atomic capture
  i = j
  j = j + 1
  !$omp end atomic

  ! CHECK-LABEL: OpenMPAtomicConstruct
  ! CHECK-NOT: OmpClause -> AcqRel
  ! CHECK: OmpClause -> Relaxed
  ! CHECK: OmpClause -> Capture
  !$omp atomic relaxed capture
  i = j
  j = j + 1
  !$omp end atomic

  ! CHECK-LABEL: OpenMPAtomicConstruct
  ! CHECK-NOT: OmpClause -> AcqRel
  ! CHECK: OmpClause -> Capture
  ! CHECK: OmpClause -> Relaxed
  !$omp atomic capture relaxed
  i = j
  j = j + 1
  !$omp end atomic
end program requires
