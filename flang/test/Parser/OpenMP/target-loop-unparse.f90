
! RUN: %flang_fc1 -fdebug-unparse -fopenmp %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp %s | FileCheck --check-prefix="PARSE-TREE" %s

! Check for parsing of loop directive

subroutine test_loop
  integer :: i, j = 1
  !PARSE-TREE: OmpBeginLoopDirective
  !PARSE-TREE-NEXT: OmpLoopDirective -> llvm::omp::Directive = loop
  !CHECK: !$omp loop
  !$omp loop
  do i=1,10
   j = j + 1
  end do
  !$omp end loop
end subroutine

subroutine test_target_loop
  integer :: i, j = 1
  !PARSE-TREE: OmpBeginLoopDirective
  !PARSE-TREE-NEXT: OmpLoopDirective -> llvm::omp::Directive = target loop
  !CHECK: !$omp target loop
  !$omp target loop
  do i=1,10
   j = j + 1
  end do
  !$omp end target loop
end subroutine

subroutine test_target_teams_loop
  integer :: i, j = 1
  !PARSE-TREE: OmpBeginLoopDirective
  !PARSE-TREE-NEXT: OmpLoopDirective -> llvm::omp::Directive = target teams loop
  !CHECK: !$omp target teams loop
  !$omp target teams loop
  do i=1,10
   j = j + 1
  end do
  !$omp end target teams loop
end subroutine

subroutine test_target_parallel_loop
  integer :: i, j = 1
  !PARSE-TREE: OmpBeginLoopDirective
  !PARSE-TREE-NEXT: OmpLoopDirective -> llvm::omp::Directive = target parallel loop
  !CHECK: !$omp target parallel loop
  !$omp target parallel loop
  do i=1,10
   j = j + 1
  end do
  !$omp end target parallel loop
end subroutine
