! RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=50 %s | \
! RUN:   FileCheck --ignore-case %s

! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=50 %s | \
! RUN:   FileCheck --check-prefix="PARSE-TREE" %s

! Check for parsing of parallel loop combined construct (OpenMP 5.0, 2.13.2)

subroutine test_parallel_loop
  integer :: i, j = 1
  !PARSE-TREE: OmpBeginLoopDirective
  !PARSE-TREE-NEXT: OmpDirectiveName -> llvm::omp::Directive = parallel loop
  !CHECK: !$omp parallel loop
  !$omp parallel loop
  do i=1,10
   j = j + 1
  end do
end subroutine

subroutine test_parallel_loop_with_end
  integer :: i, j = 1
  !PARSE-TREE: OmpBeginLoopDirective
  !PARSE-TREE-NEXT: OmpDirectiveName -> llvm::omp::Directive = parallel loop
  !CHECK: !$omp parallel loop
  !$omp parallel loop
  do i=1,10
   j = j + 1
  end do
  !PARSE-TREE: OmpEndLoopDirective
  !PARSE-TREE-NEXT: OmpDirectiveName -> llvm::omp::Directive = parallel loop
  !CHECK: !$omp end parallel loop
  !$omp end parallel loop
end subroutine

subroutine test_parallel_loop_with_clauses
  integer :: i, j = 1
  !PARSE-TREE: OmpBeginLoopDirective
  !PARSE-TREE-NEXT: OmpDirectiveName -> llvm::omp::Directive = parallel loop
  !CHECK: !$omp parallel loop num_threads(4_4) collapse(1_4) private(j) default(shared)
  !$omp parallel loop num_threads(4) collapse(1) private(j) default(shared)
  do i=1,10
   j = j + 1
  end do
end subroutine

subroutine test_parallel_loop_with_reduction
  integer :: i, total
  total = 0
  !PARSE-TREE: OmpBeginLoopDirective
  !PARSE-TREE-NEXT: OmpDirectiveName -> llvm::omp::Directive = parallel loop
  !CHECK: !$omp parallel loop reduction(+: total)
  !$omp parallel loop reduction(+:total)
  do i=1,10
   total = total + i
  end do
end subroutine

subroutine test_parallel_loop_with_if
  integer :: i, j = 1
  logical :: cond
  cond = .true.
  !PARSE-TREE: OmpBeginLoopDirective
  !PARSE-TREE-NEXT: OmpDirectiveName -> llvm::omp::Directive = parallel loop
  !CHECK: !$omp parallel loop if(cond) proc_bind(close)
  !$omp parallel loop if(cond) proc_bind(close)
  do i=1,10
   j = j + 1
  end do
end subroutine
