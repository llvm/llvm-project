!RUN: %flang_fc1 -fdebug-unparse -fopenmp %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine f00(x, y)
  implicit none
  integer :: x, y
  !$omp target map(x, y)
  x = y + 1
  y = 2 * x
  !$omp end target
end

!UNPARSE: SUBROUTINE f00 (x, y)
!UNPARSE:  IMPLICIT NONE
!UNPARSE:  INTEGER x, y
!UNPARSE: !$OMP TARGET  MAP(x,y)
!UNPARSE:   x=y+1_4
!UNPARSE:   y=2_4*x
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpBlockConstruct
!PARSE-TREE: | OmpBeginDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = target
!PARSE-TREE: | | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | OmpObject -> Designator -> DataRef -> Name = 'y'
!PARSE-TREE: | | | bool = 'true'
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'x=y+1_4'
!PARSE-TREE: | | | Variable = 'x'
!PARSE-TREE: | | | | Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | Expr = 'y+1_4'
!PARSE-TREE: | | | | Add
!PARSE-TREE: | | | | | Expr = 'y'
!PARSE-TREE: | | | | | | Designator -> DataRef -> Name = 'y'
!PARSE-TREE: | | | | | Expr = '1_4'
!PARSE-TREE: | | | | | | LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'y=2_4*x'
!PARSE-TREE: | | | Variable = 'y'
!PARSE-TREE: | | | | Designator -> DataRef -> Name = 'y'
!PARSE-TREE: | | | Expr = '2_4*x'
!PARSE-TREE: | | | | Multiply
!PARSE-TREE: | | | | | Expr = '2_4'
!PARSE-TREE: | | | | | | LiteralConstant -> IntLiteralConstant = '2'
!PARSE-TREE: | | | | | Expr = 'x'
!PARSE-TREE: | | | | | | Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | OmpEndDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = target
!PARSE-TREE: | | OmpClauseList ->


subroutine f01(x, y)
  implicit none
  integer :: x, y
  !$omp target map(x, y)
  block
    x = y + 1
    y = 2 * x
  endblock
  ! No end-directive
end

!UNPARSE: SUBROUTINE f01 (x, y)
!UNPARSE:  IMPLICIT NONE
!UNPARSE:  INTEGER x, y
!UNPARSE: !$OMP TARGET  MAP(x,y)
!UNPARSE:  BLOCK
!UNPARSE:    x=y+1_4
!UNPARSE:    y=2_4*x
!UNPARSE:  END BLOCK
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpBlockConstruct
!PARSE-TREE: | OmpBeginDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = target
!PARSE-TREE: | | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | OmpObject -> Designator -> DataRef -> Name = 'y'
!PARSE-TREE: | | | bool = 'true'
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> BlockConstruct
!PARSE-TREE: | | | BlockStmt ->
!PARSE-TREE: | | | BlockSpecificationPart -> SpecificationPart
!PARSE-TREE: | | | | ImplicitPart ->
!PARSE-TREE: | | | Block
!PARSE-TREE: | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'x=y+1_4'
!PARSE-TREE: | | | | | Variable = 'x'
!PARSE-TREE: | | | | | | Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | | | Expr = 'y+1_4'
!PARSE-TREE: | | | | | | Add
!PARSE-TREE: | | | | | | | Expr = 'y'
!PARSE-TREE: | | | | | | | | Designator -> DataRef -> Name = 'y'
!PARSE-TREE: | | | | | | | Expr = '1_4'
!PARSE-TREE: | | | | | | | | LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'y=2_4*x'
!PARSE-TREE: | | | | | Variable = 'y'
!PARSE-TREE: | | | | | | Designator -> DataRef -> Name = 'y'
!PARSE-TREE: | | | | | Expr = '2_4*x'
!PARSE-TREE: | | | | | | Multiply
!PARSE-TREE: | | | | | | | Expr = '2_4'
!PARSE-TREE: | | | | | | | | LiteralConstant -> IntLiteralConstant = '2'
!PARSE-TREE: | | | | | | | Expr = 'x'
!PARSE-TREE: | | | | | | | | Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | EndBlockStmt ->


subroutine f02(x, y)
  implicit none
  integer :: x, y
  !$omp target map(x, y)
  block
    x = y + 1
    y = 2 * x
  endblock
  ! End-directive present
  !$omp end target
end

!UNPARSE: SUBROUTINE f02 (x, y)
!UNPARSE:  IMPLICIT NONE
!UNPARSE:  INTEGER x, y
!UNPARSE: !$OMP TARGET  MAP(x,y)
!UNPARSE:  BLOCK
!UNPARSE:    x=y+1_4
!UNPARSE:    y=2_4*x
!UNPARSE:  END BLOCK
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpBlockConstruct
!PARSE-TREE: | OmpBeginDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = target
!PARSE-TREE: | | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | OmpObject -> Designator -> DataRef -> Name = 'y'
!PARSE-TREE: | | | bool = 'true'
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> BlockConstruct
!PARSE-TREE: | | | BlockStmt ->
!PARSE-TREE: | | | BlockSpecificationPart -> SpecificationPart
!PARSE-TREE: | | | | ImplicitPart ->
!PARSE-TREE: | | | Block
!PARSE-TREE: | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'x=y+1_4'
!PARSE-TREE: | | | | | Variable = 'x'
!PARSE-TREE: | | | | | | Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | | | Expr = 'y+1_4'
!PARSE-TREE: | | | | | | Add
!PARSE-TREE: | | | | | | | Expr = 'y'
!PARSE-TREE: | | | | | | | | Designator -> DataRef -> Name = 'y'
!PARSE-TREE: | | | | | | | Expr = '1_4'
!PARSE-TREE: | | | | | | | | LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'y=2_4*x'
!PARSE-TREE: | | | | | Variable = 'y'
!PARSE-TREE: | | | | | | Designator -> DataRef -> Name = 'y'
!PARSE-TREE: | | | | | Expr = '2_4*x'
!PARSE-TREE: | | | | | | Multiply
!PARSE-TREE: | | | | | | | Expr = '2_4'
!PARSE-TREE: | | | | | | | | LiteralConstant -> IntLiteralConstant = '2'
!PARSE-TREE: | | | | | | | Expr = 'x'
!PARSE-TREE: | | | | | | | | Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | EndBlockStmt ->
!PARSE-TREE: | OmpEndDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = target
!PARSE-TREE: | | OmpClauseList ->
