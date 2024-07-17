! RUN: %flang_fc1 -fdebug-unparse -fopenmp %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp %s | FileCheck --check-prefix="PARSE-TREE" %s

! Check for parsing of masked directive with filter clause. 


subroutine test_masked()
  integer :: c = 1
  !PARSE-TREE: OmpBeginBlockDirective
  !PARSE-TREE-NEXT: OmpBlockDirective -> llvm::omp::Directive = masked
  !CHECK: !$omp masked
  !$omp masked 
  c = c + 1
  !$omp end masked
  !PARSE-TREE: OmpBeginBlockDirective
  !PARSE-TREE-NEXT: OmpBlockDirective -> llvm::omp::Directive = masked
  !PARSE-TREE-NEXT: OmpClauseList -> OmpClause -> Filter -> Scalar -> Integer -> Expr = '1_4'
  !PARSE-TREE-NEXT: LiteralConstant -> IntLiteralConstant = '1'
  !CHECK: !$omp masked filter(1_4)
  !$omp masked filter(1) 
  c = c + 2
  !$omp end masked
end subroutine

subroutine test_masked_taskloop_simd()
  integer :: i, j = 1
  !PARSE-TREE: OmpBeginLoopDirective
  !PARSE-TREE-NEXT: OmpLoopDirective -> llvm::omp::Directive = masked taskloop simd
  !CHECK: !$omp masked taskloop simd
  !$omp masked taskloop simd 
  do i=1,10
   j = j + 1
  end do
  !$omp end masked taskloop simd
end subroutine

subroutine test_masked_taskloop
  integer :: i, j = 1
  !PARSE-TREE: OmpBeginLoopDirective
  !PARSE-TREE-NEXT: OmpLoopDirective -> llvm::omp::Directive = masked taskloop
  !PARSE-TREE-NEXT: OmpClauseList -> OmpClause -> Filter -> Scalar -> Integer -> Expr = '2_4'
  !PARSE-TREE-NEXT: LiteralConstant -> IntLiteralConstant = '2'
  !CHECK: !$omp masked taskloop filter(2_4)
  !$omp masked taskloop filter(2) 
  do i=1,10
   j = j + 1
  end do
  !$omp end masked taskloop 
end subroutine

subroutine test_parallel_masked
  integer, parameter :: i = 1, j = 1
  integer :: c = 2
  !PARSE-TREE: OmpBeginBlockDirective
  !PARSE-TREE-NEXT: OmpBlockDirective -> llvm::omp::Directive = parallel masked
  !PARSE-TREE-NEXT: OmpClauseList -> OmpClause -> Filter -> Scalar -> Integer -> Expr = '2_4'
  !PARSE-TREE-NEXT: Add
  !PARSE-TREE-NEXT: Expr = '1_4'
  !PARSE-TREE-NEXT: Designator -> DataRef -> Name = 'i'
  !PARSE-TREE-NEXT: Expr = '1_4'
  !PARSE-TREE-NEXT: Designator -> DataRef -> Name = 'j'
  !CHECK: !$omp parallel masked filter(2_4)
  !$omp parallel masked filter(i+j)
  c = c + 2
  !$omp end parallel masked
end subroutine

subroutine test_parallel_masked_taskloop_simd
  integer :: i, j = 1
  !PARSE-TREE: OmpBeginLoopDirective
  !PARSE-TREE-NEXT: OmpLoopDirective -> llvm::omp::Directive = parallel masked taskloop simd
  !CHECK: !$omp parallel masked taskloop simd
  !$omp parallel masked taskloop simd 
  do i=1,10
   j = j + 1
  end do
  !$omp end parallel masked taskloop simd
end subroutine

subroutine test_parallel_masked_taskloop
  integer :: i, j = 1
  !PARSE-TREE: OmpBeginLoopDirective
  !PARSE-TREE-NEXT: OmpLoopDirective -> llvm::omp::Directive = parallel masked taskloop
  !PARSE-TREE-NEXT: OmpClauseList -> OmpClause -> Filter -> Scalar -> Integer -> Expr = '2_4'
  !PARSE-TREE-NEXT: LiteralConstant -> IntLiteralConstant = '2'
  !CHECK: !$omp parallel masked taskloop filter(2_4)
  !$omp parallel masked taskloop filter(2) 
  do i=1,10
   j = j + 1
  end do
  !$omp end parallel masked taskloop 
end subroutine
