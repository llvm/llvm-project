! REQUIRES: openmp_runtime

! RUN: %flang_fc1 %openmp_flags -fdebug-dump-parse-tree %s | FileCheck %s
! Ensures associated declarative OMP allocations in the specification
! part are kept there

program allocate_tree
    use omp_lib
    integer, allocatable :: w, xarray(:), zarray(:, :)
    integer :: f
!$omp allocate(f) allocator(omp_default_mem_alloc)
    f = 2
!$omp allocate(w) allocator(omp_const_mem_alloc)
!$omp allocate(xarray) allocator(omp_large_cap_mem_alloc)
!$omp allocate(zarray) allocator(omp_default_mem_alloc)
!$omp allocate
    allocate (w, xarray(4), zarray(5, f))
end program allocate_tree

!CHECK:      | | DeclarationConstruct -> SpecificationConstruct -> OpenMPDeclarativeConstruct -> OmpAllocateDirective
!CHECK-NEXT: | | | OmpBeginDirective
!CHECK-NEXT: | | | | OmpDirectiveName -> llvm::omp::Directive = allocate
!CHECK-NEXT: | | | | OmpArgumentList -> OmpArgument -> OmpLocator -> OmpObject -> Designator -> DataRef -> Name = 'f'
!CHECK-NEXT: | | | | OmpClauseList -> OmpClause -> Allocator -> Scalar -> Integer -> Expr = '1_8'
!CHECK-NEXT: | | | | | Designator -> DataRef -> Name = 'omp_default_mem_alloc'
!CHECK-NEXT: | | | | Flags = None
!CHECK-NEXT: | | | Block
!CHECK-NEXT: | ExecutionPart -> Block
!CHECK-NEXT: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'f=2_4'
!CHECK-NEXT: | | | Variable = 'f'
!CHECK-NEXT: | | | | Designator -> DataRef -> Name = 'f'
!CHECK-NEXT: | | | Expr = '2_4'
!CHECK-NEXT: | | | | LiteralConstant -> IntLiteralConstant = '2'
!CHECK-NEXT: | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpAllocateDirective
!CHECK-NEXT: | | | OmpBeginDirective
!CHECK-NEXT: | | | | OmpDirectiveName -> llvm::omp::Directive = allocate
!CHECK-NEXT: | | | | OmpArgumentList -> OmpArgument -> OmpLocator -> OmpObject -> Designator -> DataRef -> Name = 'w'
!CHECK-NEXT: | | | | OmpClauseList -> OmpClause -> Allocator -> Scalar -> Integer -> Expr = '3_8'
!CHECK-NEXT: | | | | | Designator -> DataRef -> Name = 'omp_const_mem_alloc'
!CHECK-NEXT: | | | | Flags = None
!CHECK-NEXT: | | | Block
!CHECK-NEXT: | | | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpAllocateDirective
!CHECK-NEXT: | | | | | OmpBeginDirective
!CHECK-NEXT: | | | | | | OmpDirectiveName -> llvm::omp::Directive = allocate
!CHECK-NEXT: | | | | | | OmpArgumentList -> OmpArgument -> OmpLocator -> OmpObject -> Designator -> DataRef -> Name = 'xarray'
!CHECK-NEXT: | | | | | | OmpClauseList -> OmpClause -> Allocator -> Scalar -> Integer -> Expr = '2_8'
!CHECK-NEXT: | | | | | | | Designator -> DataRef -> Name = 'omp_large_cap_mem_alloc'
!CHECK-NEXT: | | | | | | Flags = None
!CHECK-NEXT: | | | | | Block
!CHECK-NEXT: | | | | | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpAllocateDirective
!CHECK-NEXT: | | | | | | | OmpBeginDirective
!CHECK-NEXT: | | | | | | | | OmpDirectiveName -> llvm::omp::Directive = allocate
!CHECK-NEXT: | | | | | | | | OmpArgumentList -> OmpArgument -> OmpLocator -> OmpObject -> Designator -> DataRef -> Name = 'zarray'
!CHECK-NEXT: | | | | | | | | OmpClauseList -> OmpClause -> Allocator -> Scalar -> Integer -> Expr = '1_8'
!CHECK-NEXT: | | | | | | | | | Designator -> DataRef -> Name = 'omp_default_mem_alloc'
!CHECK-NEXT: | | | | | | | | Flags = None
!CHECK-NEXT: | | | | | | | Block
!CHECK-NEXT: | | | | | | | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpAllocateDirective
!CHECK-NEXT: | | | | | | | | | OmpBeginDirective
!CHECK-NEXT: | | | | | | | | | | OmpDirectiveName -> llvm::omp::Directive = allocate
!CHECK-NEXT: | | | | | | | | | | OmpClauseList ->
!CHECK-NEXT: | | | | | | | | | | Flags = None
!CHECK-NEXT: | | | | | | | | | Block
!CHECK-NEXT: | | | | | | | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AllocateStmt
