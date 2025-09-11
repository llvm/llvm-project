!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=60 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=60 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine f00
  !$omp taskgraph
  block
  end block
end

!UNPARSE: SUBROUTINE f00
!UNPARSE: !$OMP TASKGRAPH
!UNPARSE:  BLOCK
!UNPARSE:  END BLOCK
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpBlockConstruct
!PARSE-TREE: | OmpBeginDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = taskgraph
!PARSE-TREE: | | OmpClauseList ->
!PARSE-TREE: | | Flags = None
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> BlockConstruct
!PARSE-TREE: | | | BlockStmt ->
!PARSE-TREE: | | | BlockSpecificationPart -> SpecificationPart
!PARSE-TREE: | | | | ImplicitPart ->
!PARSE-TREE: | | | Block
!PARSE-TREE: | | | EndBlockStmt ->


subroutine f01(x, y)
  integer :: x
  logical :: y
  !$omp taskgraph graph_id(x) graph_reset(y)
  !$omp task
    continue
  !$omp end task
  !$omp end taskgraph
end

!UNPARSE: SUBROUTINE f01 (x, y)
!UNPARSE:  INTEGER x
!UNPARSE:  LOGICAL y
!UNPARSE: !$OMP TASKGRAPH GRAPH_ID(x) GRAPH_RESET(y)
!UNPARSE: !$OMP TASK
!UNPARSE:  CONTINUE
!UNPARSE: !$OMP END TASK
!UNPARSE: !$OMP END TASKGRAPH
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpBlockConstruct
!PARSE-TREE: | OmpBeginDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = taskgraph
!PARSE-TREE: | | OmpClauseList -> OmpClause -> GraphId -> OmpGraphIdClause -> Expr = 'x'
!PARSE-TREE: | | | Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | OmpClause -> GraphReset -> OmpGraphResetClause -> Expr = 'y'
!PARSE-TREE: | | | Designator -> DataRef -> Name = 'y'
!PARSE-TREE: | | Flags = None
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpBlockConstruct
!PARSE-TREE: | | | OmpBeginDirective
!PARSE-TREE: | | | | OmpDirectiveName -> llvm::omp::Directive = task
!PARSE-TREE: | | | | OmpClauseList ->
!PARSE-TREE: | | | | Flags = None
!PARSE-TREE: | | | Block
!PARSE-TREE: | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> ContinueStmt
!PARSE-TREE: | | | OmpEndDirective
!PARSE-TREE: | | | | OmpDirectiveName -> llvm::omp::Directive = task
!PARSE-TREE: | | | | OmpClauseList ->
!PARSE-TREE: | | | | Flags = None
!PARSE-TREE: | OmpEndDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = taskgraph
!PARSE-TREE: | | OmpClauseList ->
!PARSE-TREE: | | Flags = None


subroutine f02
  !$omp taskgraph graph_reset
  !$omp end taskgraph
end

!UNPARSE: SUBROUTINE f02
!UNPARSE: !$OMP TASKGRAPH GRAPH_RESET
!UNPARSE: !$OMP END TASKGRAPH
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpBlockConstruct
!PARSE-TREE: | OmpBeginDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = taskgraph
!PARSE-TREE: | | OmpClauseList -> OmpClause -> GraphReset ->
!PARSE-TREE: | | Flags = None
!PARSE-TREE: | Block
!PARSE-TREE: | OmpEndDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = taskgraph
!PARSE-TREE: | | OmpClauseList ->
!PARSE-TREE: | | Flags = None
