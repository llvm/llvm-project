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

  ! CHECK-LABEL: OpenMPAtomicConstruct -> OmpAtomicRead
  ! CHECK: OmpMemoryOrderClause -> OmpClause -> Acquire
  !$omp atomic read
  i = j

  ! CHECK-LABEL: OpenMPAtomicConstruct -> OmpAtomicRead
  ! CHECK-NOT: OmpMemoryOrderClause -> OmpClause -> Acquire
  ! CHECK: OmpMemoryOrderClause -> OmpClause -> Relaxed
  !$omp atomic relaxed read
  i = j

  ! CHECK-LABEL: OpenMPAtomicConstruct -> OmpAtomicRead
  ! CHECK-NOT: OmpMemoryOrderClause -> OmpClause -> Acquire
  ! CHECK: OmpMemoryOrderClause -> OmpClause -> Relaxed
  !$omp atomic read relaxed
  i = j
  
  ! ----------------------------------------------------------------------------
  ! WRITE
  ! ----------------------------------------------------------------------------

  ! CHECK-LABEL: OpenMPAtomicConstruct -> OmpAtomicWrite
  ! CHECK: OmpMemoryOrderClause -> OmpClause -> Release
  !$omp atomic write
  i = j

  ! CHECK-LABEL: OpenMPAtomicConstruct -> OmpAtomicWrite
  ! CHECK-NOT: OmpMemoryOrderClause -> OmpClause -> Release
  ! CHECK: OmpMemoryOrderClause -> OmpClause -> Relaxed
  !$omp atomic relaxed write
  i = j

  ! CHECK-LABEL: OpenMPAtomicConstruct -> OmpAtomicWrite
  ! CHECK-NOT: OmpMemoryOrderClause -> OmpClause -> Release
  ! CHECK: OmpMemoryOrderClause -> OmpClause -> Relaxed
  !$omp atomic write relaxed
  i = j

  ! ----------------------------------------------------------------------------
  ! UPDATE
  ! ----------------------------------------------------------------------------

  ! CHECK-LABEL: OpenMPAtomicConstruct -> OmpAtomicUpdate
  ! CHECK: OmpMemoryOrderClause -> OmpClause -> Release
  !$omp atomic update
  i = i + j

  ! CHECK-LABEL: OpenMPAtomicConstruct -> OmpAtomicUpdate
  ! CHECK-NOT: OmpMemoryOrderClause -> OmpClause -> Release
  ! CHECK: OmpMemoryOrderClause -> OmpClause -> Relaxed
  !$omp atomic relaxed update
  i = i + j

  ! CHECK-LABEL: OpenMPAtomicConstruct -> OmpAtomicUpdate
  ! CHECK-NOT: OmpMemoryOrderClause -> OmpClause -> Release
  ! CHECK: OmpMemoryOrderClause -> OmpClause -> Relaxed
  !$omp atomic update relaxed
  i = i + j

  ! CHECK-LABEL: OpenMPAtomicConstruct -> OmpAtomic
  ! CHECK: OmpMemoryOrderClause -> OmpClause -> Release
  !$omp atomic
  i = i + j

  ! CHECK-LABEL: OpenMPAtomicConstruct -> OmpAtomic
  ! CHECK-NOT: OmpMemoryOrderClause -> OmpClause -> Release
  ! CHECK: OmpMemoryOrderClause -> OmpClause -> Relaxed
  !$omp atomic relaxed
  i = i + j

  ! ----------------------------------------------------------------------------
  ! CAPTURE
  ! ----------------------------------------------------------------------------

  ! CHECK-LABEL: OpenMPAtomicConstruct -> OmpAtomicCapture
  ! CHECK: OmpMemoryOrderClause -> OmpClause -> AcqRel
  !$omp atomic capture
  i = j
  j = j + 1
  !$omp end atomic

  ! CHECK-LABEL: OpenMPAtomicConstruct -> OmpAtomicCapture
  ! CHECK-NOT: OmpMemoryOrderClause -> OmpClause -> AcqRel
  ! CHECK: OmpMemoryOrderClause -> OmpClause -> Relaxed
  !$omp atomic relaxed capture
  i = j
  j = j + 1
  !$omp end atomic

  ! CHECK-LABEL: OpenMPAtomicConstruct -> OmpAtomicCapture
  ! CHECK-NOT: OmpMemoryOrderClause -> OmpClause -> AcqRel
  ! CHECK: OmpMemoryOrderClause -> OmpClause -> Relaxed
  !$omp atomic capture relaxed
  i = j
  j = j + 1
  !$omp end atomic
end program requires
