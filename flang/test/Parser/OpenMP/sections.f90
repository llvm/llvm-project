! RUN: %flang_fc1 -fdebug-unparse -fopenmp %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine openmp_sections(x, y)

  integer, intent(inout)::x, y

!==============================================================================
! empty construct
!==============================================================================
!CHECK: !$omp sections
!$omp sections
  !CHECK: !$omp section
!CHECK: !$omp end sections
!$omp end sections

!PARSE-TREE: OpenMPConstruct -> OpenMPSectionsConstruct
!PARSE-TREE: OmpBeginSectionsDirective
!PARSE-TREE-NOT: ExecutionPartConstruct
!PARSE-TREE: OmpEndSectionsDirective

!==============================================================================
! single section, without `!$omp section`
!==============================================================================
!CHECK: !$omp sections
!$omp sections
  !CHECK: !$omp section
    !CHECK: CALL
    call F1()
!CHECK: !$omp end sections
!$omp end sections

!PARSE-TREE: OpenMPConstruct -> OpenMPSectionsConstruct
!PARSE-TREE:  OmpBeginSectionsDirective
!PARSE-TREE:   OpenMPConstruct -> OpenMPSectionConstruct -> Block
!PARSE-TREE:    CallStmt
!PARSE-TREE-NOT: ExecutionPartConstruct
!PARSE-TREE:  OmpEndSectionsDirective

!==============================================================================
! single section with `!$omp section`
!==============================================================================
!CHECK: !$omp sections
!$omp sections
  !CHECK: !$omp section
  !$omp section
    !CHECK: CALL F1
    call F1
!CHECK: !$omp end sections
!$omp end sections

!PARSE-TREE: OpenMPConstruct -> OpenMPSectionsConstruct
!PARSE-TREE:  OmpBeginSectionsDirective
!PARSE-TREE:   OpenMPConstruct -> OpenMPSectionConstruct -> Block
!PARSE-TREE:    CallStmt
!PARSE-TREE-NOT: ExecutionPartConstruct
!PARSE-TREE:  OmpEndSectionsDirective

!==============================================================================
! multiple sections
!==============================================================================
!CHECK: !$omp sections
!$omp sections
  !CHECK: !$omp section
  !$omp section
    !CHECK: CALL F1
    call F1
  !CHECK: !$omp section
  !$omp section
    !CHECK: CALL F2
    call F2
  !CHECK: !$omp section
  !$omp section
    !CHECK: CALL F3
    call F3
!CHECK: !$omp end sections
!$omp end sections

!PARSE-TREE: OpenMPConstruct -> OpenMPSectionsConstruct
!PARSE-TREE:  OmpBeginSectionsDirective
!PARSE-TREE:   OpenMPConstruct -> OpenMPSectionConstruct -> Block
!PARSE-TREE:    CallStmt
!PARSE-TREE:   OpenMPConstruct -> OpenMPSectionConstruct -> Block
!PARSE-TREE:    CallStmt
!PARSE-TREE:   OpenMPConstruct -> OpenMPSectionConstruct -> Block
!PARSE-TREE:    CallStmt
!PARSE-TREE-NOT: ExecutionPartConstruct
!PARSE-TREE:  OmpEndSectionsDirective

!==============================================================================
! multiple sections with clauses
!==============================================================================
!CHECK: !$omp sections PRIVATE(x) FIRSTPRIVATE(y)
!$omp sections PRIVATE(x) FIRSTPRIVATE(y)
  !CHECK: !$omp section
  !$omp section
    !CHECK: CALL F1
    call F1
  !CHECK: !$omp section
  !$omp section
    !CHECK: CALL F2
    call F2
  !CHECK: !$omp section
  !$omp section
    !CHECK: CALL F3
    call F3
!CHECK: !$omp end sections NOWAIT
!$omp end sections NOWAIT

!PARSE-TREE: OpenMPConstruct -> OpenMPSectionsConstruct
!PARSE-TREE:  OmpBeginSectionsDirective
!PARSE-TREE:   OpenMPConstruct -> OpenMPSectionConstruct -> Block
!PARSE-TREE:    CallStmt
!PARSE-TREE:   OpenMPConstruct -> OpenMPSectionConstruct -> Block
!PARSE-TREE:    CallStmt
!PARSE-TREE:   OpenMPConstruct -> OpenMPSectionConstruct -> Block
!PARSE-TREE:    CallStmt
!PARSE-TREE-NOT: ExecutionPartConstruct
!PARSE-TREE:  OmpEndSectionsDirective

END subroutine openmp_sections
