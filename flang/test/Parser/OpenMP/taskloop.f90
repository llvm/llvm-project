! RUN: %flang_fc1 -fdebug-unparse -fopenmp %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine parallel_work
  integer :: i

!CHECK: !$OMP TASKLOOP  GRAINSIZE(STRICT: 500_4)
!PARSE-TREE: OmpBeginLoopDirective
!PARSE-TREE-NEXT: OmpLoopDirective -> llvm::omp::Directive = taskloop
!PARSE-TREE-NEXT: OmpClauseList -> OmpClause -> Grainsize -> OmpGrainsizeClause
!PARSE-TREE-NEXT: Modifier -> OmpPrescriptiveness -> Value = Strict
!PARSE-TREE-NEXT: Scalar -> Integer -> Expr = '500_4'
  !$omp taskloop grainsize(strict: 500)
  do i=1,10000
    call loop_body(i)
  end do
  !$omp end taskloop

!CHECK: !$OMP TASKLOOP  GRAINSIZE(500_4)
!PARSE-TREE: OmpBeginLoopDirective
!PARSE-TREE-NEXT: OmpLoopDirective -> llvm::omp::Directive = taskloop
!PARSE-TREE-NEXT: OmpClauseList -> OmpClause -> Grainsize -> OmpGrainsizeClause
!PARSE-TREE-NEXT: Scalar -> Integer -> Expr = '500_4'
  !$omp taskloop grainsize(500)
  do i=1,10000
    call loop_body(i)
  end do
  !$omp end taskloop

!CHECK: !$OMP TASKLOOP  NUM_TASKS(STRICT: 500_4)
!PARSE-TREE: OmpBeginLoopDirective
!PARSE-TREE-NEXT: OmpLoopDirective -> llvm::omp::Directive = taskloop
!PARSE-TREE-NEXT: OmpClauseList -> OmpClause -> NumTasks -> OmpNumTasksClause
!PARSE-TREE-NEXT: Modifier -> OmpPrescriptiveness -> Value = Strict
!PARSE-TREE-NEXT: Scalar -> Integer -> Expr = '500_4'
  !$omp taskloop num_tasks(strict: 500)
  do i=1,10000
    call loop_body(i)
  end do
  !$omp end taskloop
end subroutine parallel_work
