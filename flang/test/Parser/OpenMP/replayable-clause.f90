!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=60 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=60 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine f00
  !$omp task replayable
  block
  end block
end

!UNPARSE: SUBROUTINE f00
!UNPARSE: !$OMP TASK REPLAYABLE
!UNPARSE:  BLOCK
!UNPARSE:  END BLOCK
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpBlockConstruct
!PARSE-TREE: | OmpBeginDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = task
!PARSE-TREE: | | OmpClauseList -> OmpClause -> Replayable ->
!PARSE-TREE: | | Flags = None
!PARSE-TREE: | Block


subroutine f01(x)
  implicit none
  integer :: x
  !$omp target_update to(x) replayable(.true.)
end

!UNPARSE: SUBROUTINE f01 (x)
!UNPARSE:  IMPLICIT NONE
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP TARGET_UPDATE TO(x) REPLAYABLE(.true._4)
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPSimpleStandaloneConstruct -> OmpDirectiveSpecification
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = target update
!PARSE-TREE: | OmpClauseList -> OmpClause -> To -> OmpToClause
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | bool = 'true'
!PARSE-TREE: | OmpClause -> Replayable -> OmpReplayableClause -> Scalar -> Logical -> Constant -> Expr = '.true._4'
!PARSE-TREE: | | LiteralConstant -> LogicalLiteralConstant
!PARSE-TREE: | | | bool = 'true'
!PARSE-TREE: | Flags = None


subroutine f02
  !$omp taskwait replayable(.false.)
end

!UNPARSE: SUBROUTINE f02
!UNPARSE: !$OMP TASKWAIT REPLAYABLE(.false._4)
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPSimpleStandaloneConstruct -> OmpDirectiveSpecification
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = taskwait
!PARSE-TREE: | OmpClauseList -> OmpClause -> Replayable -> OmpReplayableClause -> Scalar -> Logical -> Constant -> Expr = '.false._4'
!PARSE-TREE: | | LiteralConstant -> LogicalLiteralConstant
!PARSE-TREE: | | | bool = 'false'
!PARSE-TREE: | Flags = None
