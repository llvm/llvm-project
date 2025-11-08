!RUN: %flang_fc1 -fdebug-unparse-no-sema -fopenmp %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema -fopenmp %s | FileCheck --check-prefix="PARSE-TREE" %s

! Check that constructs A and B are parsed correctly, where the name of A
! is a prefix of B.
! Currently it's TARGET vs TARGET DATA, TARGET ENTER DATA, TARGET EXIT DATA,
! and TARGET UPDATE.

subroutine f00(x)
  implicit none
  integer :: x
  !$omp target
  !$omp target data map(x)
  x = x + 1
  !$omp end target data
  !$omp end target
end

!UNPARSE: SUBROUTINE f00 (x)
!UNPARSE:  IMPLICIT NONE
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP TARGET
!UNPARSE: !$OMP TARGET DATA  MAP(x)
!UNPARSE:   x = x+1
!UNPARSE: !$OMP END TARGET DATA
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpBlockConstruct
!PARSE-TREE: | OmpBeginDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = target
!PARSE-TREE: | | OmpClauseList ->
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpBlockConstruct
!PARSE-TREE: | | | OmpBeginDirective
!PARSE-TREE: | | | | OmpDirectiveName -> llvm::omp::Directive = target data
!PARSE-TREE: | | | | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | | | bool = 'true'
!PARSE-TREE: | | | Block
!PARSE-TREE: | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt
!PARSE-TREE: | | | | | Variable -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | | | Expr -> Add
!PARSE-TREE: | | | | | | Expr -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | | | | Expr -> LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | | | OmpEndDirective
!PARSE-TREE: | | | | OmpDirectiveName -> llvm::omp::Directive = target data
!PARSE-TREE: | | | | OmpClauseList ->
!PARSE-TREE: | OmpEndDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = target
!PARSE-TREE: | | OmpClauseList ->


subroutine f01(x)
  implicit none
  integer :: x
  !$omp target
  !$omp target enter data map(x)
  x = x + 1
  !$omp end target
end

!UNPARSE: SUBROUTINE f01 (x)
!UNPARSE:  IMPLICIT NONE
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP TARGET
!UNPARSE: !$OMP TARGET ENTER DATA MAP(x)
!UNPARSE:   x = x+1
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpBlockConstruct
!PARSE-TREE: | OmpBeginDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = target
!PARSE-TREE: | | OmpClauseList ->
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPSimpleStandaloneConstruct -> OmpDirectiveSpecification
!PARSE-TREE: | | | OmpDirectiveName -> llvm::omp::Directive = target enter data
!PARSE-TREE: | | | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | | bool = 'true'
!PARSE-TREE: | | | Flags = None
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt
!PARSE-TREE: | | | Variable -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | Expr -> Add
!PARSE-TREE: | | | | Expr -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | | Expr -> LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | OmpEndDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = target
!PARSE-TREE: | | OmpClauseList ->


subroutine f02(x)
  implicit none
  integer :: x
  !$omp target
  !$omp target exit data map(x)
  x = x + 1
  !$omp end target
end

!UNPARSE: SUBROUTINE f02 (x)
!UNPARSE:  IMPLICIT NONE
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP TARGET
!UNPARSE: !$OMP TARGET EXIT DATA MAP(x)
!UNPARSE:   x = x+1
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpBlockConstruct
!PARSE-TREE: | OmpBeginDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = target
!PARSE-TREE: | | OmpClauseList ->
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPSimpleStandaloneConstruct -> OmpDirectiveSpecification
!PARSE-TREE: | | | OmpDirectiveName -> llvm::omp::Directive = target exit data
!PARSE-TREE: | | | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | | bool = 'true'
!PARSE-TREE: | | | Flags = None
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt
!PARSE-TREE: | | | Variable -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | Expr -> Add
!PARSE-TREE: | | | | Expr -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | | Expr -> LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | OmpEndDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = target
!PARSE-TREE: | | OmpClauseList ->


subroutine f03(x)
  implicit none
  integer :: x
  !$omp target
  !$omp target update to(x)
  x = x + 1
  !$omp end target
end

!UNPARSE: SUBROUTINE f03 (x)
!UNPARSE:  IMPLICIT NONE
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP TARGET
!UNPARSE: !$OMP TARGET UPDATE TO(x)
!UNPARSE:   x = x+1
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpBlockConstruct
!PARSE-TREE: | OmpBeginDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = target
!PARSE-TREE: | | OmpClauseList ->
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPSimpleStandaloneConstruct -> OmpDirectiveSpecification
!PARSE-TREE: | | | OmpDirectiveName -> llvm::omp::Directive = target update
!PARSE-TREE: | | | OmpClauseList -> OmpClause -> To -> OmpToClause
!PARSE-TREE: | | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | | bool = 'true'
!PARSE-TREE: | | | Flags = None
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt
!PARSE-TREE: | | | Variable -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | Expr -> Add
!PARSE-TREE: | | | | Expr -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | | Expr -> LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | OmpEndDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = target
!PARSE-TREE: | | OmpClauseList ->
