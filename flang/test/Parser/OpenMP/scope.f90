! RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=51 %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=51 %s | FileCheck --check-prefix="PARSE-TREE" %s

program omp_scope
  integer i
  i = 10

!CHECK: !$OMP SCOPE  PRIVATE(i)
!CHECK: !$OMP END SCOPE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpBlockConstruct
!PARSE-TREE: OmpBeginDirective
!PARSE-TREE: OmpDirectiveName -> llvm::omp::Directive = scope
!PARSE-TREE: OmpClauseList -> OmpClause -> Private -> OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'i'
!PARSE-TREE: Block
!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> PrintStmt
!PARSE-TREE: OmpEndDirective
!PARSE-TREE: OmpDirectiveName -> llvm::omp::Directive = scope
!PARSE-TREE: OmpClauseList -> OmpClause -> Nowait

  !$omp scope private(i)
  print *, "omp scope", i
  !$omp end scope nowait
end program omp_scope
