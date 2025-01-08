! RUN: %flang_fc1 -fdebug-unparse -fopenmp %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp %s | FileCheck --check-prefix="PARSE-TREE" %s

! Check for parsing of master directive


subroutine test_master()
  integer :: c = 1
  !PARSE-TREE: OmpBeginBlockDirective
  !PARSE-TREE-NEXT: OmpBlockDirective -> llvm::omp::Directive = master
  !CHECK: !$omp master
  !$omp master 
  c = c + 1
  !$omp end master
end subroutine

subroutine test_master_taskloop_simd()
  integer :: i, j = 1
  !PARSE-TREE: OmpBeginLoopDirective
  !PARSE-TREE-NEXT: OmpLoopDirective -> llvm::omp::Directive = master taskloop simd
  !CHECK: !$omp master taskloop simd
  !$omp master taskloop simd 
  do i=1,10
   j = j + 1
  end do
  !$omp end master taskloop simd
end subroutine

subroutine test_master_taskloop
  integer :: i, j = 1
  !PARSE-TREE: OmpBeginLoopDirective
  !PARSE-TREE-NEXT: OmpLoopDirective -> llvm::omp::Directive = master taskloop
  !CHECK: !$omp master taskloop
  !$omp master taskloop
  do i=1,10
   j = j + 1
  end do
  !$omp end master taskloop 
end subroutine

subroutine test_parallel_master
  integer :: c = 2
  !PARSE-TREE: OmpBeginBlockDirective
  !PARSE-TREE-NEXT: OmpBlockDirective -> llvm::omp::Directive = parallel master
  !CHECK: !$omp parallel master
  !$omp parallel master
  c = c + 2
  !$omp end parallel master
end subroutine

subroutine test_parallel_master_taskloop_simd
  integer :: i, j = 1
  !PARSE-TREE: OmpBeginLoopDirective
  !PARSE-TREE-NEXT: OmpLoopDirective -> llvm::omp::Directive = parallel master taskloop simd
  !CHECK: !$omp parallel master taskloop simd
  !$omp parallel master taskloop simd 
  do i=1,10
   j = j + 1
  end do
  !$omp end parallel master taskloop simd
end subroutine

subroutine test_parallel_master_taskloop
  integer :: i, j = 1
  !PARSE-TREE: OmpBeginLoopDirective
  !PARSE-TREE-NEXT: OmpLoopDirective -> llvm::omp::Directive = parallel master taskloop
  !CHECK: !$omp parallel master taskloop
  !$omp parallel master taskloop
  do i=1,10
   j = j + 1
  end do
  !$omp end parallel master taskloop 
end subroutine
