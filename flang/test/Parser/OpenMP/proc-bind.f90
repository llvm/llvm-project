! RUN: %flang_fc1 -fdebug-unparse -fopenmp %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp %s | FileCheck --check-prefix="PARSE-TREE" %s

! CHECK: !$OMP PARALLEL  PROC_BIND(PRIMARY)

! PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpBlockConstruct
! PARSE-TREE:  OmpBeginDirective
! PARSE-TREE:   OmpDirectiveName -> llvm::omp::Directive = parallel
! PARSE-TREE:   OmpClauseList -> OmpClause -> ProcBind -> OmpProcBindClause -> AffinityPolicy = Primary
subroutine sb1
  !$omp parallel proc_bind(primary)
  print *, "Hello"
  !$omp end parallel
end subroutine
