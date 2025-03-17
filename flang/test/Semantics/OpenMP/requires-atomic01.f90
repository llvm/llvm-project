! RUN: %flang_fc1 -fopenmp -fopenmp-version=50 -fdebug-dump-parse-tree %s 2>&1 | FileCheck %s
! Ensure that requires atomic_default_mem_order is used to update atomic
! operations with no explicit memory order set.
program requires
  implicit none
  !$omp requires atomic_default_mem_order(seq_cst)
  integer :: i, j

  ! ----------------------------------------------------------------------------
  ! READ
  ! ----------------------------------------------------------------------------

  ! CHECK-LABEL: OpenMPAtomicConstruct
  ! CHECK: OmpClause -> Read
  ! CHECK: OmpClause -> SeqCst
  !$omp atomic read
  i = j

  ! CHECK-LABEL: OpenMPAtomicConstruct
  ! CHECK-NOT: OmpClause -> SeqCst
  ! CHECK: OmpClause -> Relaxed
  ! CHECK: OmpClause -> Read
  !$omp atomic relaxed read
  i = j

  ! CHECK-LABEL: OpenMPAtomicConstruct
  ! CHECK: OmpClause -> Read
  ! CHECK-NOT: OmpClause -> SeqCst
  ! CHECK: OmpClause -> Relaxed
  !$omp atomic read relaxed
  i = j
  
  ! ----------------------------------------------------------------------------
  ! WRITE
  ! ----------------------------------------------------------------------------

  ! CHECK-LABEL: OpenMPAtomicConstruct
  ! CHECK: OmpClause -> Write
  ! CHECK: OmpClause -> SeqCst
  !$omp atomic write
  i = j

  ! CHECK-LABEL: OpenMPAtomicConstruct
  ! CHECK-NOT: OmpClause -> SeqCst
  ! CHECK: OmpClause -> Relaxed
  ! CHECK: OmpClause -> Write
  !$omp atomic relaxed write
  i = j

  ! CHECK-LABEL: OpenMPAtomicConstruct
  ! CHECK: OmpClause -> Write
  ! CHECK-NOT: OmpClause -> SeqCst
  ! CHECK: OmpClause -> Relaxed
  !$omp atomic write relaxed
  i = j

  ! ----------------------------------------------------------------------------
  ! UPDATE
  ! ----------------------------------------------------------------------------

  ! CHECK-LABEL: OpenMPAtomicConstruct
  ! CHECK: OmpClause -> Update
  ! CHECK: OmpClause -> SeqCst
  !$omp atomic update
  i = i + j

  ! CHECK-LABEL: OpenMPAtomicConstruct
  ! CHECK-NOT: OmpClause -> SeqCst
  ! CHECK: OmpClause -> Relaxed
  ! CHECK: OmpClause -> Update
  !$omp atomic relaxed update
  i = i + j

  ! CHECK-LABEL: OpenMPAtomicConstruct
  ! CHECK: OmpClause -> Update
  ! CHECK-NOT: OmpClause -> SeqCst
  ! CHECK: OmpClause -> Relaxed
  !$omp atomic update relaxed
  i = i + j

  ! CHECK-LABEL: OpenMPAtomicConstruct
  ! CHECK: OmpClause -> SeqCst
  !$omp atomic
  i = i + j

  ! CHECK-LABEL: OpenMPAtomicConstruct
  ! CHECK-NOT: OmpClause -> SeqCst
  ! CHECK: OmpClause -> Relaxed
  !$omp atomic relaxed
  i = i + j

  ! ----------------------------------------------------------------------------
  ! CAPTURE
  ! ----------------------------------------------------------------------------

  ! CHECK-LABEL: OpenMPAtomicConstruct
  ! CHECK: OmpClause -> Capture
  ! CHECK: OmpClause -> SeqCst
  !$omp atomic capture
  i = j
  j = j + 1
  !$omp end atomic

  ! CHECK-LABEL: OpenMPAtomicConstruct
  ! CHECK-NOT: OmpClause -> SeqCst
  ! CHECK: OmpClause -> Relaxed
  ! CHECK: OmpClause -> Capture
  !$omp atomic relaxed capture
  i = j
  j = j + 1
  !$omp end atomic

  ! CHECK-LABEL: OpenMPAtomicConstruct
  ! CHECK: OmpClause -> Capture
  ! CHECK-NOT: OmpClause -> SeqCst
  ! CHECK: OmpClause -> Relaxed
  !$omp atomic capture relaxed
  i = j
  j = j + 1
  !$omp end atomic
end program requires
