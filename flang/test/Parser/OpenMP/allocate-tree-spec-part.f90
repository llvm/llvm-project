! REQUIRES: openmp_runtime

! RUN: %flang_fc1 -fopenmp -fdebug-dump-parse-tree %s | FileCheck %s
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

!CHECK: | | DeclarationConstruct -> SpecificationConstruct -> OpenMPDeclarativeConstruct -> OpenMPDeclarativeAllocate
!CHECK-NEXT: | | | Verbatim
!CHECK-NEXT: | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'f'
!CHECK-NEXT: | | | OmpClauseList -> OmpClause -> Allocator -> Scalar -> Integer -> Expr =
!CHECK-NEXT: | | | | Designator -> DataRef -> Name =
!CHECK-NEXT: | ExecutionPart -> Block
!CHECK-NEXT: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'f=2_4'
!CHECK-NEXT: | | | Variable = 'f'
!CHECK-NEXT: | | | | Designator -> DataRef -> Name = 'f'
!CHECK-NEXT: | | | Expr = '2_4'
!CHECK-NEXT: | | | | LiteralConstant -> IntLiteralConstant = '2'
!CHECK-NEXT: | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPExecutableAllocate
!CHECK-NEXT: | | | Verbatim
!CHECK-NEXT: | | | OmpClauseList ->
!CHECK-NEXT: | | | OpenMPDeclarativeAllocate
!CHECK-NEXT: | | | | Verbatim
!CHECK-NEXT: | | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'w'
!CHECK-NEXT: | | | | OmpClauseList -> OmpClause -> Allocator -> Scalar -> Integer -> Expr =
!CHECK-NEXT: | | | | | Designator -> DataRef -> Name =
!CHECK-NEXT: | | | OpenMPDeclarativeAllocate
!CHECK-NEXT: | | | | Verbatim
!CHECK-NEXT: | | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'xarray'
!CHECK-NEXT: | | | | OmpClauseList -> OmpClause -> Allocator -> Scalar -> Integer -> Expr =
!CHECK-NEXT: | | | | | Designator -> DataRef -> Name =
!CHECK-NEXT: | | | OpenMPDeclarativeAllocate
!CHECK-NEXT: | | | | Verbatim
!CHECK-NEXT: | | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'zarray'
!CHECK-NEXT: | | | | OmpClauseList -> OmpClause -> Allocator -> Scalar -> Integer -> Expr =
!CHECK-NEXT: | | | | | Designator -> DataRef -> Name =
!CHECK-NEXT: | | | AllocateStmt
