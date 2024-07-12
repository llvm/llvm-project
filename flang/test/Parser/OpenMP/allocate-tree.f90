! REQUIRES: openmp_runtime

! RUN: %flang_fc1 %openmp_flags -fdebug-dump-parse-tree %s | FileCheck %s
! RUN: %flang_fc1 %openmp_flags -fdebug-unparse %s | FileCheck %s --check-prefix="UNPARSE"
! Ensures associated declarative OMP allocations are nested in their
! corresponding executable allocate directive

program allocate_tree
    use omp_lib
    integer, allocatable :: w, xarray(:), zarray(:, :)
    integer :: z, t
    t = 2
    z = 3
!$omp allocate(w) allocator(omp_const_mem_alloc)
!$omp allocate(xarray) allocator(omp_large_cap_mem_alloc)
!$omp allocate(zarray) allocator(omp_default_mem_alloc)
!$omp allocate
    allocate(w, xarray(4), zarray(t, z))
end program allocate_tree

!CHECK: | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPExecutableAllocate
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

!UNPARSE: !$OMP ALLOCATE (w) ALLOCATOR(3_8)
!UNPARSE-NEXT: !$OMP ALLOCATE (xarray) ALLOCATOR(2_8)
!UNPARSE-NEXT: !$OMP ALLOCATE (zarray) ALLOCATOR(1_8)
!UNPARSE-NEXT: !$OMP ALLOCATE
!UNPARSE-NEXT: ALLOCATE(w, xarray(4_4), zarray(t,z))
