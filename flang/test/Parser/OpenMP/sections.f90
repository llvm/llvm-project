! RUN: %flang_fc1 -fdebug-unparse -fopenmp %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine openmp_sections(x, y)

  integer, intent(inout)::x, y

!==============================================================================
! empty construct
!==============================================================================
!CHECK: !$omp sections
!$omp sections
!CHECK: !$omp end sections
!$omp end sections

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPSectionsConstruct
!PARSE-TREE: | OmpBeginSectionsDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = sections
!PARSE-TREE: | | OmpClauseList ->
!PARSE-TREE: | | Flags = None
!PARSE-TREE: | OpenMPConstruct -> OpenMPSectionConstruct
!PARSE-TREE: | | Block
!PARSE-TREE: | OmpEndSectionsDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = sections
!PARSE-TREE: | | OmpClauseList ->
!PARSE-TREE: | | Flags = None

!==============================================================================
! single section, without `!$omp section`
!==============================================================================
!CHECK: !$omp sections
!$omp sections
    !CHECK: CALL
    call F1()
!CHECK: !$omp end sections
!$omp end sections

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPSectionsConstruct
!PARSE-TREE: | OmpBeginSectionsDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = sections
!PARSE-TREE: | | OmpClauseList ->
!PARSE-TREE: | | Flags = None
!PARSE-TREE: | OpenMPConstruct -> OpenMPSectionConstruct
!PARSE-TREE: | | Block
!PARSE-TREE: | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> CallStmt = 'CALL f1()'
!PARSE-TREE: | | | | Call
!PARSE-TREE: | | | | | ProcedureDesignator -> Name = 'f1'
!PARSE-TREE: | OmpEndSectionsDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = sections
!PARSE-TREE: | | OmpClauseList ->
!PARSE-TREE: | | Flags = None

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

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPSectionsConstruct
!PARSE-TREE: | OmpBeginSectionsDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = sections
!PARSE-TREE: | | OmpClauseList ->
!PARSE-TREE: | | Flags = None
!PARSE-TREE: | OpenMPConstruct -> OpenMPSectionConstruct
!PARSE-TREE: | | OmpDirectiveSpecification
!PARSE-TREE: | | | OmpDirectiveName -> llvm::omp::Directive = section
!PARSE-TREE: | | | OmpClauseList ->
!PARSE-TREE: | | | Flags = None
!PARSE-TREE: | | Block
!PARSE-TREE: | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> CallStmt = 'CALL f1()'
!PARSE-TREE: | | | | Call
!PARSE-TREE: | | | | | ProcedureDesignator -> Name = 'f1'
!PARSE-TREE: | OmpEndSectionsDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = sections
!PARSE-TREE: | | OmpClauseList ->
!PARSE-TREE: | | Flags = None

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

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPSectionsConstruct
!PARSE-TREE: | OmpBeginSectionsDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = sections
!PARSE-TREE: | | OmpClauseList ->
!PARSE-TREE: | | Flags = None
!PARSE-TREE: | OpenMPConstruct -> OpenMPSectionConstruct
!PARSE-TREE: | | OmpDirectiveSpecification
!PARSE-TREE: | | | OmpDirectiveName -> llvm::omp::Directive = section
!PARSE-TREE: | | | OmpClauseList ->
!PARSE-TREE: | | | Flags = None
!PARSE-TREE: | | Block
!PARSE-TREE: | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> CallStmt = 'CALL f1()'
!PARSE-TREE: | | | | Call
!PARSE-TREE: | | | | | ProcedureDesignator -> Name = 'f1'
!PARSE-TREE: | OpenMPConstruct -> OpenMPSectionConstruct
!PARSE-TREE: | | OmpDirectiveSpecification
!PARSE-TREE: | | | OmpDirectiveName -> llvm::omp::Directive = section
!PARSE-TREE: | | | OmpClauseList ->
!PARSE-TREE: | | | Flags = None
!PARSE-TREE: | | Block
!PARSE-TREE: | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> CallStmt = 'CALL f2()'
!PARSE-TREE: | | | | Call
!PARSE-TREE: | | | | | ProcedureDesignator -> Name = 'f2'
!PARSE-TREE: | OpenMPConstruct -> OpenMPSectionConstruct
!PARSE-TREE: | | OmpDirectiveSpecification
!PARSE-TREE: | | | OmpDirectiveName -> llvm::omp::Directive = section
!PARSE-TREE: | | | OmpClauseList ->
!PARSE-TREE: | | | Flags = None
!PARSE-TREE: | | Block
!PARSE-TREE: | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> CallStmt = 'CALL f3()'
!PARSE-TREE: | | | | Call
!PARSE-TREE: | | | | | ProcedureDesignator -> Name = 'f3'
!PARSE-TREE: | OmpEndSectionsDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = sections
!PARSE-TREE: | | OmpClauseList ->
!PARSE-TREE: | | Flags = None

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

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPSectionsConstruct
!PARSE-TREE: | OmpBeginSectionsDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = sections
!PARSE-TREE: | | OmpClauseList -> OmpClause -> Private -> OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | OmpClause -> Firstprivate -> OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'y'
!PARSE-TREE: | | Flags = None
!PARSE-TREE: | OpenMPConstruct -> OpenMPSectionConstruct
!PARSE-TREE: | | OmpDirectiveSpecification
!PARSE-TREE: | | | OmpDirectiveName -> llvm::omp::Directive = section
!PARSE-TREE: | | | OmpClauseList ->
!PARSE-TREE: | | | Flags = None
!PARSE-TREE: | | Block
!PARSE-TREE: | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> CallStmt = 'CALL f1()'
!PARSE-TREE: | | | | Call
!PARSE-TREE: | | | | | ProcedureDesignator -> Name = 'f1'
!PARSE-TREE: | OpenMPConstruct -> OpenMPSectionConstruct
!PARSE-TREE: | | OmpDirectiveSpecification
!PARSE-TREE: | | | OmpDirectiveName -> llvm::omp::Directive = section
!PARSE-TREE: | | | OmpClauseList ->
!PARSE-TREE: | | | Flags = None
!PARSE-TREE: | | Block
!PARSE-TREE: | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> CallStmt = 'CALL f2()'
!PARSE-TREE: | | | | Call
!PARSE-TREE: | | | | | ProcedureDesignator -> Name = 'f2'
!PARSE-TREE: | OpenMPConstruct -> OpenMPSectionConstruct
!PARSE-TREE: | | OmpDirectiveSpecification
!PARSE-TREE: | | | OmpDirectiveName -> llvm::omp::Directive = section
!PARSE-TREE: | | | OmpClauseList ->
!PARSE-TREE: | | | Flags = None
!PARSE-TREE: | | Block
!PARSE-TREE: | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> CallStmt = 'CALL f3()'
!PARSE-TREE: | | | | Call
!PARSE-TREE: | | | | | ProcedureDesignator -> Name = 'f3'
!PARSE-TREE: | OmpEndSectionsDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = sections
!PARSE-TREE: | | OmpClauseList -> OmpClause -> Nowait
!PARSE-TREE: | | Flags = None

END subroutine openmp_sections
