! RUN: %flang_fc1 -fdebug-unparse-no-sema -fopenmp %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema -fopenmp %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine foo()
  integer :: i, j
  j = 0
! CHECK: !$OMP DO  REDUCTION(TASK, *: j)
! PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
! PARSE-TREE: | | | OmpBeginLoopDirective
! PARSE-TREE: | | | | OmpLoopDirective -> llvm::omp::Directive = do
! PARSE-TREE: | | | | OmpClauseList -> OmpClause -> Reduction -> OmpReductionClause
! PARSE-TREE: | | | | | Modifier -> OmpReductionModifier -> Value = Task
! PARSE-TREE: | | | | | Modifier -> OmpReductionIdentifier -> DefinedOperator -> IntrinsicOperator = Multiply
! PARSE-TREE: | | | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'j
  !$omp do reduction (task, *: j)
  do i = 1, 10
    j = j + 1
  end do
  !$omp end do
end
