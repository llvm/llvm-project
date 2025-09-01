!RUN: %flang_fc1 -fopenmp-version=51 -fopenmp -fdebug-unparse-no-sema %s | FileCheck --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fopenmp-version=51 -fopenmp -fdebug-dump-parse-tree-no-sema %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine sub1
  integer :: r
  !$omp assume no_openmp
  !$omp end assume

  !$omp assume no_parallelism
  !$omp end assume

  !$omp assume no_openmp_routines
  !$omp end assume

  !$omp assume absent(allocate), contains(workshare, task)
  block ! strictly-structured-block
  end block

  !$omp assume holds(1.eq.1)
  block
  end block
  print *, r
end subroutine sub1

!UNPARSE: SUBROUTINE sub1
!UNPARSE:  INTEGER r
!UNPARSE: !$OMP ASSUME NO_OPENMP
!UNPARSE: !$OMP END ASSUME
!UNPARSE: !$OMP ASSUME NO_PARALLELISM
!UNPARSE: !$OMP END ASSUME
!UNPARSE: !$OMP ASSUME NO_OPENMP_ROUTINES
!UNPARSE: !$OMP END ASSUME
!UNPARSE: !$OMP ASSUME ABSENT(ALLOCATE) CONTAINS(WORKSHARE,TASK)
!UNPARSE:  BLOCK
!UNPARSE:  END BLOCK
!UNPARSE: !$OMP ASSUME HOLDS(1==1)
!UNPARSE:  BLOCK
!UNPARSE:  END BLOCK
!UNPARSE:  PRINT *, r
!UNPARSE: END SUBROUTINE sub1

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPAssumeConstruct
!PARSE-TREE: | OmpBeginDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = assume
!PARSE-TREE: | | OmpClauseList -> OmpClause -> NoOpenmp
!PARSE-TREE: | | Flags = None
!PARSE-TREE: | Block
!PARSE-TREE: | OmpEndDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = assume
!PARSE-TREE: | | OmpClauseList ->
!PARSE-TREE: | | Flags = None
!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPAssumeConstruct
!PARSE-TREE: | OmpBeginDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = assume
!PARSE-TREE: | | OmpClauseList -> OmpClause -> NoParallelism
!PARSE-TREE: | | Flags = None
!PARSE-TREE: | Block
!PARSE-TREE: | OmpEndDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = assume
!PARSE-TREE: | | OmpClauseList ->
!PARSE-TREE: | | Flags = None
!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPAssumeConstruct
!PARSE-TREE: | OmpBeginDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = assume
!PARSE-TREE: | | OmpClauseList -> OmpClause -> NoOpenmpRoutines
!PARSE-TREE: | | Flags = None
!PARSE-TREE: | Block
!PARSE-TREE: | OmpEndDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = assume
!PARSE-TREE: | | OmpClauseList ->
!PARSE-TREE: | | Flags = None
!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPAssumeConstruct
!PARSE-TREE: | OmpBeginDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = assume
!PARSE-TREE: | | OmpClauseList -> OmpClause -> Absent -> OmpAbsentClause -> llvm::omp::Directive = allocate
!PARSE-TREE: | | OmpClause -> Contains -> OmpContainsClause -> llvm::omp::Directive = workshare
!PARSE-TREE: | | llvm::omp::Directive = task
!PARSE-TREE: | | Flags = None
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> BlockConstruct
!PARSE-TREE: | | | BlockStmt ->
!PARSE-TREE: | | | BlockSpecificationPart -> SpecificationPart
!PARSE-TREE: | | | | ImplicitPart ->
!PARSE-TREE: | | | Block
!PARSE-TREE: | | | EndBlockStmt ->
!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPAssumeConstruct
!PARSE-TREE: | OmpBeginDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = assume
!PARSE-TREE: | | OmpClauseList -> OmpClause -> Holds -> OmpHoldsClause -> Expr -> EQ
!PARSE-TREE: | | | Expr -> LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | | | Expr -> LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | | Flags = None
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> BlockConstruct
!PARSE-TREE: | | | BlockStmt ->
!PARSE-TREE: | | | BlockSpecificationPart -> SpecificationPart
!PARSE-TREE: | | | | ImplicitPart ->
!PARSE-TREE: | | | Block
!PARSE-TREE: | | | EndBlockStmt ->


subroutine sub2
  integer :: r
  integer :: v
  v = 87
  !$omp assume no_openmp
  r = r + 1
  !$omp end assume
end subroutine sub2

!UNPARSE: SUBROUTINE sub2
!UNPARSE:  INTEGER r
!UNPARSE:  INTEGER v
!UNPARSE:  v = 87
!UNPARSE: !$OMP ASSUME NO_OPENMP
!UNPARSE:  r = r+1
!UNPARSE: !$OMP END ASSUME
!UNPARSE: END SUBROUTINE sub2

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt
!PARSE-TREE: | Variable -> Designator -> DataRef -> Name = 'v'
!PARSE-TREE: | Expr -> LiteralConstant -> IntLiteralConstant = '87'
!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPAssumeConstruct
!PARSE-TREE: | OmpBeginDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = assume
!PARSE-TREE: | | OmpClauseList -> OmpClause -> NoOpenmp
!PARSE-TREE: | | Flags = None
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt
!PARSE-TREE: | | | Variable -> Designator -> DataRef -> Name = 'r'
!PARSE-TREE: | | | Expr -> Add
!PARSE-TREE: | | | | Expr -> Designator -> DataRef -> Name = 'r'
!PARSE-TREE: | | | | Expr -> LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | OmpEndDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = assume
!PARSE-TREE: | | OmpClauseList ->
!PARSE-TREE: | | Flags = None

program p
  !$omp assumes no_openmp
end program p

!UNPARSE: PROGRAM p
!UNPARSE: !$OMP ASSUMES  NO_OPENMP
!UNPARSE: END PROGRAM p

!PARSE-TREE: OpenMPDeclarativeConstruct -> OpenMPDeclarativeAssumes
!PARSE-TREE: | Verbatim
!PARSE-TREE: | OmpClauseList -> OmpClause -> NoOpenmp
