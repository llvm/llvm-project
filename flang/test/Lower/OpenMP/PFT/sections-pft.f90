! RUN: %flang_fc1 -fdebug-pre-fir-tree -fopenmp %s | FileCheck %s

subroutine openmp_sections(x, y)

  integer, intent(inout)::x, y

!==============================================================================
! empty construct
!==============================================================================
!$omp sections
!$omp end sections

!CHECK: OpenMPConstruct
!CHECK: End OpenMPConstruct

!==============================================================================
! single section, without `!$omp section`
!==============================================================================
!$omp sections
    call F1()
!$omp end sections

!CHECK: OpenMPConstruct
!CHECK:  OpenMPConstruct
!CHECK:   CallStmt
!CHECK:  End OpenMPConstruct
!CHECK: End OpenMPConstruct

!==============================================================================
! single section with `!$omp section`
!==============================================================================
!$omp sections
  !$omp section
    call F1
!$omp end sections

!CHECK: OpenMPConstruct
!CHECK:  OpenMPConstruct
!CHECK:   CallStmt
!CHECK:  End OpenMPConstruct
!CHECK: End OpenMPConstruct

!==============================================================================
! multiple sections
!==============================================================================
!$omp sections
  !$omp section
    call F1
  !$omp section
    call F2
  !$omp section
    call F3
!$omp end sections

!CHECK: OpenMPConstruct
!CHECK:  OpenMPConstruct
!CHECK:   CallStmt
!CHECK:  End OpenMPConstruct
!CHECK:  OpenMPConstruct
!CHECK:   CallStmt
!CHECK:  End OpenMPConstruct
!CHECK:  OpenMPConstruct
!CHECK:   CallStmt
!CHECK:  End OpenMPConstruct
!CHECK: End OpenMPConstruct

!==============================================================================
! multiple sections with clauses
!==============================================================================
!$omp sections PRIVATE(x) FIRSTPRIVATE(y)
  !$omp section
    call F1
  !$omp section
    call F2
  !$omp section
    call F3
!$omp end sections NOWAIT

!CHECK: OpenMPConstruct
!CHECK:  OpenMPConstruct
!CHECK:   CallStmt
!CHECK:  End OpenMPConstruct
!CHECK:  OpenMPConstruct
!CHECK:   CallStmt
!CHECK:  End OpenMPConstruct
!CHECK:  OpenMPConstruct
!CHECK:   CallStmt
!CHECK:  End OpenMPConstruct
!CHECK: End OpenMPConstruct

end subroutine openmp_sections
