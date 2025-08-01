!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=50 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=50 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine f00
  integer :: x
!$omp taskgroup task_reduction(+: x)
  x = x + 1
!$omp end taskgroup
end

!UNPARSE: SUBROUTINE f00
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP TASKGROUP  TASK_REDUCTION(+: x)
!UNPARSE:   x=x+1_4
!UNPARSE: !$OMP END TASKGROUP
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: | OmpBlockDirective -> llvm::omp::Directive = taskgroup
!PARSE-TREE: | OmpClauseList -> OmpClause -> TaskReduction -> OmpTaskReductionClause
!PARSE-TREE: | | Modifier -> OmpReductionIdentifier -> DefinedOperator -> IntrinsicOperator = Add
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: Block
