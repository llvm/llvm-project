! REQUIRES: openmp_runtime

! RUN: %flang_fc1 %openmp_flags -fopenmp-version=51 -fdebug-dump-parse-tree %s | FileCheck %s
! RUN: %flang_fc1 %openmp_flags -fdebug-unparse -fopenmp-version=51 %s | FileCheck %s --check-prefix="UNPARSE"
! Ensures associated declarative OMP allocations are nested in their
! corresponding executable allocate directive

program allocate_align_tree
    use omp_lib
    integer, allocatable :: j(:), xarray(:)
    integer :: z, t
    t = 2
    z = 3
!$omp allocate(j) align(16)
!$omp allocate(xarray) align(32) allocator(omp_large_cap_mem_alloc)
    allocate(j(z), xarray(t))
end program allocate_align_tree

!CHECK: | | DeclarationConstruct -> SpecificationConstruct -> TypeDeclarationStmt
!CHECK-NEXT: | | | DeclarationTypeSpec -> IntrinsicTypeSpec -> IntegerTypeSpec ->
!CHECK-NEXT: | | | AttrSpec -> Allocatable
!CHECK-NEXT: | | | EntityDecl
!CHECK-NEXT: | | | | Name = 'j'


!CHECK: | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPExecutableAllocate
!CHECK-NEXT: | | | Verbatim
!CHECK-NEXT: | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'xarray'
!CHECK-NEXT: | | | OmpClauseList -> OmpClause -> Align -> OmpAlignClause -> Scalar -> Integer -> Constant -> Expr = '32_4'
!CHECK-NEXT: | | | | LiteralConstant -> IntLiteralConstant = '32'
!CHECK-NEXT: | | | OmpClause -> Allocator -> Scalar -> Integer -> Expr = '2_8'
!CHECK-NEXT: | | | | Designator -> DataRef -> Name = 'omp_large_cap_mem_alloc'
!CHECK-NEXT: | | | OpenMPDeclarativeAllocate
!CHECK-NEXT: | | | | Verbatim
!CHECK-NEXT: | | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'j'
!CHECK-NEXT: | | | | OmpClauseList -> OmpClause -> Align -> OmpAlignClause -> Scalar -> Integer -> Constant -> Expr = '16_4'
!CHECK-NEXT: | | | | | LiteralConstant -> IntLiteralConstant = '16'
!CHECK-NEXT: | | | AllocateStmt

!UNPARSE: !$OMP ALLOCATE (j) ALIGN(16_4)
!UNPARSE: !$OMP ALLOCATE (xarray) ALIGN(32_4) ALLOCATOR(2_8)
!UNPARSE-NEXT: ALLOCATE(j(z), xarray(t))
