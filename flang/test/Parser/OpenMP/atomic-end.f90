!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=60 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=60 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine f00
  integer :: x, v
  !$omp atomic read
  v = x
  !$omp end atomic
end

!UNPARSE: SUBROUTINE f00
!UNPARSE:  INTEGER x, v
!UNPARSE: !$OMP ATOMIC READ
!UNPARSE:   v=x
!UNPARSE: !$OMP END ATOMIC
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPAtomicConstruct
!PARSE-TREE: | OmpDirectiveSpecification
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = atomic
!PARSE-TREE: | | OmpClauseList -> OmpClause -> Read
!PARSE-TREE: | | Flags = None
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'v=x'
!PARSE-TREE: | | | Variable = 'v'
!PARSE-TREE: | | | | Designator -> DataRef -> Name = 'v'
!PARSE-TREE: | | | Expr = 'x'
!PARSE-TREE: | | | | Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | OmpDirectiveSpecification
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = atomic
!PARSE-TREE: | | OmpClauseList ->
!PARSE-TREE: | | Flags = None


subroutine f01
  integer :: x, v
  !$omp atomic read
  v = x
  !$omp endatomic
end

!UNPARSE: SUBROUTINE f01
!UNPARSE:  INTEGER x, v
!UNPARSE: !$OMP ATOMIC READ
!UNPARSE:   v=x
!UNPARSE: !$OMP END ATOMIC
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPAtomicConstruct
!PARSE-TREE: | OmpDirectiveSpecification
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = atomic
!PARSE-TREE: | | OmpClauseList -> OmpClause -> Read
!PARSE-TREE: | | Flags = None
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'v=x'
!PARSE-TREE: | | | Variable = 'v'
!PARSE-TREE: | | | | Designator -> DataRef -> Name = 'v'
!PARSE-TREE: | | | Expr = 'x'
!PARSE-TREE: | | | | Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | OmpDirectiveSpecification
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = atomic
!PARSE-TREE: | | OmpClauseList ->
!PARSE-TREE: | | Flags = None
