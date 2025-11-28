! REQUIRES: openmp_runtime

! RUN: %flang_fc1 %openmp_flags -fdebug-dump-parse-tree %s | FileCheck %s
! RUN: %flang_fc1 %openmp_flags -fdebug-unparse %s | FileCheck %s --check-prefix="UNPARSE"
! Ensures associated declarative OMP allocations are nested in their
! corresponding executable allocate directive

program allocate_tree
    use omp_lib
    integer, allocatable :: xarray(:), zarray(:, :)
    integer :: z, t, w
!$omp allocate(w) allocator(omp_const_mem_alloc)
    t = 2
    z = 3
!$omp allocate(xarray) allocator(omp_large_cap_mem_alloc)
!$omp allocate(zarray) allocator(omp_default_mem_alloc)
!$omp allocate
    allocate(xarray(4), zarray(t, z))
end program allocate_tree

!CHECK:      DeclarationConstruct -> SpecificationConstruct -> OpenMPDeclarativeConstruct -> OmpAllocateDirective
!CHECK-NEXT: | OmpBeginDirective
!CHECK-NEXT: | | OmpDirectiveName -> llvm::omp::Directive = allocate
!CHECK-NEXT: | | OmpArgumentList -> OmpArgument -> OmpLocator -> OmpObject -> Designator -> DataRef -> Name = 'w'
!CHECK-NEXT: | | OmpClauseList -> OmpClause -> Allocator -> Scalar -> Integer -> Expr = '3_8'
!CHECK-NEXT: | | | Designator -> DataRef -> Name = 'omp_const_mem_alloc'
!CHECK-NEXT: | | Flags = None
!CHECK-NEXT: | Block

!CHECK:      ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpAllocateDirective
!CHECK-NEXT: | OmpBeginDirective
!CHECK-NEXT: | | OmpDirectiveName -> llvm::omp::Directive = allocate
!CHECK-NEXT: | | OmpArgumentList -> OmpArgument -> OmpLocator -> OmpObject -> Designator -> DataRef -> Name = 'xarray'
!CHECK-NEXT: | | OmpClauseList -> OmpClause -> Allocator -> Scalar -> Integer -> Expr = '2_8'
!CHECK-NEXT: | | | Designator -> DataRef -> Name = 'omp_large_cap_mem_alloc'
!CHECK-NEXT: | | Flags = None
!CHECK-NEXT: | Block
!CHECK-NEXT: | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpAllocateDirective
!CHECK-NEXT: | | | OmpBeginDirective
!CHECK-NEXT: | | | | OmpDirectiveName -> llvm::omp::Directive = allocate
!CHECK-NEXT: | | | | OmpArgumentList -> OmpArgument -> OmpLocator -> OmpObject -> Designator -> DataRef -> Name = 'zarray'
!CHECK-NEXT: | | | | OmpClauseList -> OmpClause -> Allocator -> Scalar -> Integer -> Expr = '1_8'
!CHECK-NEXT: | | | | | Designator -> DataRef -> Name = 'omp_default_mem_alloc'
!CHECK-NEXT: | | | | Flags = None
!CHECK-NEXT: | | | Block
!CHECK-NEXT: | | | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpAllocateDirective
!CHECK-NEXT: | | | | | OmpBeginDirective
!CHECK-NEXT: | | | | | | OmpDirectiveName -> llvm::omp::Directive = allocate
!CHECK-NEXT: | | | | | | OmpClauseList ->
!CHECK-NEXT: | | | | | | Flags = None
!CHECK-NEXT: | | | | | Block
!CHECK-NEXT: | | | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AllocateStmt

!UNPARSE:      !$OMP ALLOCATE(w) ALLOCATOR(3_8)
!UNPARSE-NEXT:   t=2_4
!UNPARSE-NEXT:   z=3_4
!UNPARSE-NEXT: !$OMP ALLOCATE(xarray) ALLOCATOR(2_8)
!UNPARSE-NEXT: !$OMP ALLOCATE(zarray) ALLOCATOR(1_8)
!UNPARSE-NEXT: !$OMP ALLOCATE
!UNPARSE-NEXT:  ALLOCATE(xarray(4_4), zarray(t,z))
