! REQUIRES: openmp_runtime

! RUN: %flang_fc1 %openmp_flags -fopenmp-version=51 -fdebug-dump-parse-tree %s | FileCheck %s
! RUN: %flang_fc1 %openmp_flags -fdebug-unparse -fopenmp-version=51 %s | FileCheck %s --check-prefix="UNPARSE"
! Ensures associated declarative OMP allocations are nested in their
! corresponding executable allocate directive

program allocate_tree
    use omp_lib
    integer, allocatable :: j
!$omp allocate(j) align(16)    
    allocate(j)
end program allocate_tree

!CHECK: | | DeclarationConstruct -> SpecificationConstruct -> TypeDeclarationStmt
!CHECK-NEXT: | | | DeclarationTypeSpec -> IntrinsicTypeSpec -> IntegerTypeSpec ->
!CHECK-NEXT: | | | AttrSpec -> Allocatable
!CHECK-NEXT: | | | EntityDecl
!CHECK-NEXT: | | | | Name = 'j'


!CHECK: | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPExecutableAllocate
!CHECK-NEXT: | | | Verbatim
!CHECK-NEXT: | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'j'
!CHECK-NEXT: | | | OmpClauseList -> OmpClause -> Align -> OmpAlignClause -> Scalar -> Integer -> Expr = '16_4'
!CHECK-NEXT: | | | | LiteralConstant -> IntLiteralConstant = '16'
!CHECK-NEXT: | | | AllocateStmt

!UNPARSE: !$OMP ALLOCATE (j) ALIGN(16_4)
!UNPARSE-NEXT: ALLOCATE(j)
