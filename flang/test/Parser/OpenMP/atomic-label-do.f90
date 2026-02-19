!RUN: %flang_fc1 -fdebug-unparse -fopenmp %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine f
  integer :: i, x
  do 100 i = 1, 10
  !$omp atomic write
  100 x = i
end

!UNPARSE: SUBROUTINE f
!UNPARSE:  INTEGER i, x
!UNPARSE:  DO i=1_4,10_4
!UNPARSE: !$OMP ATOMIC WRITE
!UNPARSE:   100  x=i
!UNPARSE:  END DO
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct
!PARSE-TREE: | NonLabelDoStmt
!PARSE-TREE: | | LoopControl -> LoopBounds
!PARSE-TREE: | | | Scalar -> Name = 'i'
!PARSE-TREE: | | | Scalar -> Expr = '1_4'
!PARSE-TREE: | | | | LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | | | Scalar -> Expr = '10_4'
!PARSE-TREE: | | | | LiteralConstant -> IntLiteralConstant = '10'
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPAtomicConstruct
!PARSE-TREE: | | | OmpBeginDirective
!PARSE-TREE: | | | | OmpDirectiveName -> llvm::omp::Directive = atomic
!PARSE-TREE: | | | | OmpClauseList -> OmpClause -> Write
!PARSE-TREE: | | | | Flags = {CrossesLabelDo}
!PARSE-TREE: | | | Block
!PARSE-TREE: | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'x=i'
!PARSE-TREE: | | | | | Variable = 'x'
!PARSE-TREE: | | | | | | Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | | | Expr = 'i'
!PARSE-TREE: | | | | | | Designator -> DataRef -> Name = 'i'
!PARSE-TREE: | EndDoStmt ->
