! REQUIRES: openmp_runtime

! RUN: %flang_fc1 %openmp_flags -fopenmp-version=52 -fdebug-unparse-no-sema %s | FileCheck %s --check-prefix=UNPARSE
! RUN: %flang_fc1 %openmp_flags -fopenmp-version=52 -fdebug-dump-parse-tree %s | FileCheck %s --check-prefix=PARSE-TREE

! Both surface syntaxes of the USES_ALLOCATORS clause must round-trip through
! the unparser, and the deprecated form must remain distinguishable from the
! OpenMP 5.2 form in the parse tree.

subroutine uses_allocators_syntax
  use omp_lib
  integer(omp_allocator_handle_kind) :: a, b
  type(omp_alloctrait), parameter :: tr(1) = &
      [omp_alloctrait(omp_atk_alignment, 64)]
  integer :: x

  !$omp target uses_allocators(omp_default_mem_alloc)
  x = 1
  !$omp end target

  !$omp target uses_allocators(traits(tr): a)
  x = 2
  !$omp end target

  !$omp target uses_allocators(memspace(omp_default_mem_space): a)
  x = 3
  !$omp end target

  !$omp target uses_allocators(memspace(omp_low_lat_mem_space), traits(tr): a)
  x = 4
  !$omp end target

  !$omp target uses_allocators(traits(tr), memspace(omp_low_lat_mem_space): a)
  x = 5
  !$omp end target

  !$omp target uses_allocators(a(tr))
  x = 6
  !$omp end target

  !$omp target uses_allocators(a(tr), b(tr))
  x = 7
  !$omp end target
end subroutine

!UNPARSE: SUBROUTINE uses_allocators_syntax
!UNPARSE: !$OMP TARGET  USES_ALLOCATORS(omp_default_mem_alloc)
!UNPARSE: !$OMP TARGET  USES_ALLOCATORS(TRAITS(tr): a)
!UNPARSE: !$OMP TARGET  USES_ALLOCATORS(MEMSPACE(omp_default_mem_space): a)
!UNPARSE: !$OMP TARGET  USES_ALLOCATORS(MEMSPACE(omp_low_lat_mem_space), TRAITS(tr): a)
!UNPARSE: !$OMP TARGET  USES_ALLOCATORS(TRAITS(tr), MEMSPACE(omp_low_lat_mem_space): a)
!UNPARSE: !$OMP TARGET  USES_ALLOCATORS(a(tr))
!UNPARSE: !$OMP TARGET  USES_ALLOCATORS(a(tr), b(tr))

! A bare allocator is the canonical 5.2 form: no modifier is recorded, and the
! specification is not flagged as the legacy syntax.
!PARSE-TREE: OmpClause -> UsesAllocators -> OmpUsesAllocatorsClause -> AllocatorSpec
!PARSE-TREE-NEXT: Scalar -> Integer -> Expr = '1_8'
!PARSE-TREE-NEXT: Designator -> DataRef -> Name = 'omp_default_mem_alloc'
!PARSE-TREE-NEXT: bool = 'false'

! TRAITS(tr): a
!PARSE-TREE: OmpClause -> UsesAllocators -> OmpUsesAllocatorsClause -> AllocatorSpec
!PARSE-TREE-NEXT: Modifier -> OmpTraitsArray -> Expr = '[omp_alloctrait::omp_alloctrait(key=2_4,value=64_8)]'
!PARSE-TREE-NEXT: Designator -> DataRef -> Name = 'tr'
!PARSE-TREE-NEXT: Scalar -> Integer -> Expr = 'a'
!PARSE-TREE-NEXT: Designator -> DataRef -> Name = 'a'
!PARSE-TREE-NEXT: bool = 'false'

! MEMSPACE(omp_default_mem_space): a
!PARSE-TREE: OmpClause -> UsesAllocators -> OmpUsesAllocatorsClause -> AllocatorSpec
!PARSE-TREE-NEXT: Modifier -> OmpMemSpace -> Scalar -> Integer -> Expr = '99_8'
!PARSE-TREE-NEXT: Designator -> DataRef -> Name = 'omp_default_mem_space'
!PARSE-TREE-NEXT: Scalar -> Integer -> Expr = 'a'
!PARSE-TREE-NEXT: Designator -> DataRef -> Name = 'a'
!PARSE-TREE-NEXT: bool = 'false'

! Both modifiers, in either order.
!PARSE-TREE: OmpClause -> UsesAllocators -> OmpUsesAllocatorsClause -> AllocatorSpec
!PARSE-TREE-NEXT: Modifier -> OmpMemSpace -> Scalar -> Integer -> Expr = '4_8'
!PARSE-TREE-NEXT: Designator -> DataRef -> Name = 'omp_low_lat_mem_space'
!PARSE-TREE-NEXT: Modifier -> OmpTraitsArray -> Expr = '[omp_alloctrait::omp_alloctrait(key=2_4,value=64_8)]'
!PARSE-TREE-NEXT: Designator -> DataRef -> Name = 'tr'
!PARSE-TREE-NEXT: Scalar -> Integer -> Expr = 'a'
!PARSE-TREE-NEXT: Designator -> DataRef -> Name = 'a'
!PARSE-TREE-NEXT: bool = 'false'

!PARSE-TREE: OmpClause -> UsesAllocators -> OmpUsesAllocatorsClause -> AllocatorSpec
!PARSE-TREE-NEXT: Modifier -> OmpTraitsArray -> Expr = '[omp_alloctrait::omp_alloctrait(key=2_4,value=64_8)]'
!PARSE-TREE-NEXT: Designator -> DataRef -> Name = 'tr'
!PARSE-TREE-NEXT: Modifier -> OmpMemSpace -> Scalar -> Integer -> Expr = '4_8'
!PARSE-TREE-NEXT: Designator -> DataRef -> Name = 'omp_low_lat_mem_space'
!PARSE-TREE-NEXT: Scalar -> Integer -> Expr = 'a'
!PARSE-TREE-NEXT: Designator -> DataRef -> Name = 'a'
!PARSE-TREE-NEXT: bool = 'false'

! The deprecated "allocator(traits)" form is stored as a traits modifier, and
! is flagged so that it unparses back to the syntax that was written.
!PARSE-TREE: OmpClause -> UsesAllocators -> OmpUsesAllocatorsClause -> AllocatorSpec
!PARSE-TREE-NEXT: Modifier -> OmpTraitsArray -> Expr = '[omp_alloctrait::omp_alloctrait(key=2_4,value=64_8)]'
!PARSE-TREE-NEXT: Designator -> DataRef -> Name = 'tr'
!PARSE-TREE-NEXT: Scalar -> Integer -> Expr = 'a'
!PARSE-TREE-NEXT: Designator -> DataRef -> Name = 'a'
!PARSE-TREE-NEXT: bool = 'true'

! The deprecated syntax can list more than one allocator specification.
!PARSE-TREE: OmpClause -> UsesAllocators -> OmpUsesAllocatorsClause -> AllocatorSpec
!PARSE-TREE-NEXT: Modifier -> OmpTraitsArray -> Expr = '[omp_alloctrait::omp_alloctrait(key=2_4,value=64_8)]'
!PARSE-TREE-NEXT: Designator -> DataRef -> Name = 'tr'
!PARSE-TREE-NEXT: Scalar -> Integer -> Expr = 'a'
!PARSE-TREE-NEXT: Designator -> DataRef -> Name = 'a'
!PARSE-TREE-NEXT: bool = 'true'
!PARSE-TREE-NEXT: AllocatorSpec
!PARSE-TREE-NEXT: Modifier -> OmpTraitsArray -> Expr = '[omp_alloctrait::omp_alloctrait(key=2_4,value=64_8)]'
!PARSE-TREE-NEXT: Designator -> DataRef -> Name = 'tr'
!PARSE-TREE-NEXT: Scalar -> Integer -> Expr = 'b'
!PARSE-TREE-NEXT: Designator -> DataRef -> Name = 'b'
!PARSE-TREE-NEXT: bool = 'true'
