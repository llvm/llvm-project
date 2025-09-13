!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=60 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=60 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine f00(x)
  implicit none
  integer :: x
  !$omp target_data map(to: x) transparent
  block
  end block
end

!UNPARSE: SUBROUTINE f00 (x)
!UNPARSE:  IMPLICIT NONE
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP TARGET_DATA MAP(TO: x) TRANSPARENT
!UNPARSE:  BLOCK
!UNPARSE:  END BLOCK
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpBlockConstruct
!PARSE-TREE: | OmpBeginDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = target data
!PARSE-TREE: | | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | | Modifier -> OmpMapType -> Value = To
!PARSE-TREE: | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | bool = 'true'
!PARSE-TREE: | | OmpClause -> Transparent ->
!PARSE-TREE: | | Flags = None
!PARSE-TREE: | Block


subroutine f01
  !$omp task transparent(0)
  !$omp end task
end

!UNPARSE: SUBROUTINE f01
!UNPARSE: !$OMP TASK TRANSPARENT(0_4)
!UNPARSE: !$OMP END TASK
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpBlockConstruct
!PARSE-TREE: | OmpBeginDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = task
!PARSE-TREE: | | OmpClauseList -> OmpClause -> Transparent -> OmpTransparentClause -> Scalar -> Integer -> Expr = '0_4'
!PARSE-TREE: | | | LiteralConstant -> IntLiteralConstant = '0'
!PARSE-TREE: | | Flags = None
!PARSE-TREE: | Block
!PARSE-TREE: | OmpEndDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = task
!PARSE-TREE: | | OmpClauseList ->
!PARSE-TREE: | | Flags = None


subroutine f02
  implicit none
  integer :: i
  !$omp taskloop transparent(2)
  do i = 1, 10
  end do
end

!UNPARSE: SUBROUTINE f02
!UNPARSE:  IMPLICIT NONE
!UNPARSE:  INTEGER i
!UNPARSE: !$OMP TASKLOOP  TRANSPARENT(2_4)
!UNPARSE:  DO i=1_4,10_4
!UNPARSE:  END DO
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE: | OmpBeginLoopDirective
!PARSE-TREE: | | OmpLoopDirective -> llvm::omp::Directive = taskloop
!PARSE-TREE: | | OmpClauseList -> OmpClause -> Transparent -> OmpTransparentClause -> Scalar -> Integer -> Expr = '2_4'
!PARSE-TREE: | | | LiteralConstant -> IntLiteralConstant = '2'
!PARSE-TREE: | DoConstruct
