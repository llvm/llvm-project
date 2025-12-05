!RUN: %flang_fc1 -fdebug-unparse -fopenmp %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine f00
  integer :: i, j
  do 100 i = 1,10
!$omp do
    do 100 j = 1,10
    100 continue
end

!UNPARSE:  SUBROUTINE f00
!UNPARSE:   INTEGER i, j
!UNPARSE:   DO i=1_4,10_4
!UNPARSE: !$OMP DO
!UNPARSE:    DO j=1_4,10_4
!UNPARSE:     100 CONTINUE
!UNPARSE:    END DO
!UNPARSE:   END DO
!UNPARSE:  END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct
!PARSE-TREE: | NonLabelDoStmt
!PARSE-TREE: | | LoopControl -> LoopBounds
!PARSE-TREE: | | | Scalar -> Name = 'i'
!PARSE-TREE: | | | Scalar -> Expr = '1_4'
!PARSE-TREE: | | | | LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | | | Scalar -> Expr = '10_4'
!PARSE-TREE: | | | | LiteralConstant -> IntLiteralConstant = '10'
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE: | | | OmpBeginLoopDirective
!PARSE-TREE: | | | | OmpDirectiveName -> llvm::omp::Directive = do
!PARSE-TREE: | | | | OmpClauseList ->
!PARSE-TREE: | | | | Flags = {CrossesLabelDo}
!PARSE-TREE: | | | Block
!PARSE-TREE: | | | | ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct
!PARSE-TREE: | | | | | NonLabelDoStmt
!PARSE-TREE: | | | | | | LoopControl -> LoopBounds
!PARSE-TREE: | | | | | | | Scalar -> Name = 'j'
!PARSE-TREE: | | | | | | | Scalar -> Expr = '1_4'
!PARSE-TREE: | | | | | | | | LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | | | | | | | Scalar -> Expr = '10_4'
!PARSE-TREE: | | | | | | | | LiteralConstant -> IntLiteralConstant = '10'
!PARSE-TREE: | | | | | Block
!PARSE-TREE: | | | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> ContinueStmt
!PARSE-TREE: | | | | | EndDoStmt ->
!PARSE-TREE: | EndDoStmt ->
