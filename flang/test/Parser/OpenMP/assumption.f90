! RUN: %flang_fc1  -fopenmp-version=51 -fopenmp -fdebug-unparse-no-sema %s 2>&1 | FileCheck %s
! RUN: %flang_fc1  -fopenmp-version=51 -fopenmp -fdebug-dump-parse-tree-no-sema %s 2>&1 | FileCheck %s --check-prefix="PARSE-TREE"
subroutine sub1
  integer :: r
!CHECK: !$OMP ASSUME NO_OPENMP
!PARSE-TREE:   ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPAssumeConstruct
!PARSE-TREE:   Verbatim
!PARSE-TREE:   OmpClauseList -> OmpClause -> NoOpenmp
  !$omp assume no_openmp
!CHECK: !$OMP ASSUME NO_PARALLELISM
!PARSE-TREE:   ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPAssumeConstruct
!PARSE-TREE:   Verbatim
!PARSE-TREE:   OmpClauseList -> OmpClause -> NoParallelism
  !$omp assume no_parallelism
!CHECK: !$OMP ASSUME NO_OPENMP_ROUTINES
!PARSE-TREE:   ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPAssumeConstruct
!PARSE-TREE:   Verbatim
!PARSE-TREE:   OmpClauseList -> OmpClause -> NoOpenmpRoutines  
  !$omp assume no_openmp_routines
!CHECK: !$OMP ASSUME ABSENT(ALLOCATE), CONTAINS(WORKSHARE,TASK)
!PARSE-TREE:   ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPAssumeConstruct
!PARSE-TREE:   Verbatim
!PARSE-TREE:   OmpClauseList -> OmpClause -> Absent -> OmpAbsentClause -> llvm::omp::Directive = allocate
!PARSE-TREE:   OmpClause -> Contains -> OmpContainsClause -> llvm::omp::Directive = workshare
!PARSE-TREE:   llvm::omp::Directive = task
 !$omp assume absent(allocate), contains(workshare, task)
!CHECK: !$OMP ASSUME HOLDS(1==1)
  !$omp assume holds(1.eq.1)
  print *, r
end subroutine sub1

subroutine sub2
  integer :: r
  integer :: v
!CHECK !$OMP ASSUME NO_OPENMP
!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPAssumeConstruct
!PARSE-TREE: OmpAssumeDirective
!PARSE-TREE: Verbatim
!PARSE-TREE: OmpClauseList -> OmpClause -> NoOpenmp
!PARSE-TREE: Block
!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt
!PARSE-TREE: Expr -> Add
!PARSE-TREE: OmpEndAssumeDirective
  v = 87
  !$omp assume no_openmp
  r = r + 1
!CHECK !$OMP END ASSUME
  !$omp end assume
end subroutine sub2
  
program p
!CHECK !$OMP ASSUMES NO_OPENMP
!PARSE-TREE: SpecificationPart
!PARSE-TREE:   OpenMPDeclarativeConstruct -> OpenMPDeclarativeAssumes
!PARSE-TREE: Verbatim
!PARSE-TREE: OmpClauseList -> OmpClause -> NoOpenmp
  !$omp assumes no_openmp
end program p
  
