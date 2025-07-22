! RUN: %flang_fc1 -fdebug-unparse-no-sema -fopenmp %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema -fopenmp %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine sub0
!CHECK: !$OMP DECLARE VARIANT (sub:vsub) MATCH(CONSTRUCT={PARALLEL})
!PARSE-TREE: OpenMPDeclarativeConstruct -> OmpDeclareVariantDirective
!PARSE-TREE: | Verbatim
!PARSE-TREE: | Name = 'sub'
!PARSE-TREE: | Name = 'vsub'
!PARSE-TREE: | OmpClauseList -> OmpClause -> Match -> OmpMatchClause -> OmpContextSelectorSpecification -> OmpTraitSetSelector
!PARSE-TREE: | | OmpTraitSetSelectorName -> Value = Construct
!PARSE-TREE: | | OmpTraitSelector
!PARSE-TREE: | | | OmpTraitSelectorName -> llvm::omp::Directive = parallel
  !$omp declare variant (sub:vsub) match (construct={parallel})
contains
  subroutine vsub
  end subroutine

  subroutine sub ()
  end subroutine
end subroutine

subroutine sb1
  integer :: x
  x = 1
  !$omp dispatch device(1)
    call sub(x)
contains
  subroutine vsub (v1)
    integer, value :: v1
  end
  subroutine sub (v1)
!CHECK: !$OMP DECLARE VARIANT (vsub) MATCH(CONSTRUCT={DISPATCH}
!PARSE-TREE: OpenMPDeclarativeConstruct -> OmpDeclareVariantDirective
!PARSE-TREE: | Verbatim
!PARSE-TREE: | Name = 'vsub'
!PARSE-TREE: | OmpClauseList -> OmpClause -> Match -> OmpMatchClause -> OmpContextSelectorSpecification -> OmpTraitSetSelector
!PARSE-TREE: | | OmpTraitSetSelectorName -> Value = Construct
!PARSE-TREE: | | OmpTraitSelector
!PARSE-TREE: | | | OmpTraitSelectorName -> llvm::omp::Directive = dispatch
    !$omp declare variant(vsub), match(construct={dispatch})
    integer, value :: v1
  end
end subroutine

subroutine sb2 (x1, x2)
  use omp_lib, only: omp_interop_kind
  integer :: x
  x = 1
  !$omp dispatch device(1)
    call sub(x)
contains
  subroutine vsub (v1, a1, a2)
    integer, value :: v1
    integer(omp_interop_kind) :: a1
    integer(omp_interop_kind), value :: a2
  end
  subroutine sub (v1)
!CHECK: !$OMP DECLARE VARIANT (vsub) MATCH(CONSTRUCT={DISPATCH}) APPEND_ARGS(INTEROP(T&
!CHECK: !$OMP&ARGET),INTEROP(TARGET))
!PARSE-TREE: OpenMPDeclarativeConstruct -> OmpDeclareVariantDirective
!PARSE-TREE: | Verbatim
!PARSE-TREE: | Name = 'vsub'
!PARSE-TREE: | OmpClauseList -> OmpClause -> Match -> OmpMatchClause -> OmpContextSelectorSpecification -> OmpTraitSetSelector
!PARSE-TREE: | | OmpTraitSetSelectorName -> Value = Construct
!PARSE-TREE: | | OmpTraitSelector
!PARSE-TREE: | | | OmpTraitSelectorName -> llvm::omp::Directive = dispatch
!PARSE-TREE: | OmpClause -> AppendArgs -> OmpAppendArgsClause -> OmpAppendOp -> OmpInteropType -> Value = Target
!PARSE-TREE: | OmpAppendOp -> OmpInteropType -> Value = Target
    !$omp declare variant(vsub), match(construct={dispatch}), append_args (interop(target), interop(target))
    integer, value :: v1
  end
end subroutine

subroutine sb3 (x1, x2)
  use iso_c_binding, only: c_ptr
  type(c_ptr), value :: x1, x2

  !$omp dispatch device(1)
  call sub(x1, x2)
contains
  subroutine sub (v1, v2)
    type(c_ptr), value :: v1, v2
!CHECK: !$OMP DECLARE VARIANT (vsub) MATCH(CONSTRUCT={DISPATCH}) ADJUST_ARGS(NOTHING:v&
!CHECK: !$OMP&1) ADJUST_ARGS(NEED_DEVICE_PTR:v2)
!PARSE-TREE: DeclarationConstruct -> SpecificationConstruct -> OpenMPDeclarativeConstruct -> OmpDeclareVariantDirective
!PARSE-TREE: | Verbatim
!PARSE-TREE: | Name = 'vsub'
!PARSE-TREE: | OmpClauseList -> OmpClause -> Match -> OmpMatchClause -> OmpContextSelectorSpecification -> OmpTraitSetSelector
!PARSE-TREE: | | OmpTraitSetSelectorName -> Value = Construct
!PARSE-TREE: | | OmpTraitSelector
!PARSE-TREE: | | | OmpTraitSelectorName -> llvm::omp::Directive = dispatch
!PARSE-TREE: | OmpClause -> AdjustArgs -> OmpAdjustArgsClause
!PARSE-TREE: | | OmpAdjustOp -> Value = Nothing
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'v1'
!PARSE-TREE: | OmpClause -> AdjustArgs -> OmpAdjustArgsClause
!PARSE-TREE: | | OmpAdjustOp -> Value = Need_Device_Ptr
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'v2'
    !$omp declare variant(vsub) match ( construct = { dispatch } ) adjust_args(nothing : v1 ) adjust_args(need_device_ptr : v2)
  end
  subroutine vsub(v1, v2)
    type(c_ptr), value :: v1, v2
  end
end subroutine

subroutine f
  real :: x, y
  y = 2
  !omp simd
  call f2(x, y)
  !omp end simd 
contains
  subroutine f1 (x, y)
    real :: x, y
  end

  subroutine f2 (x, y)
    real :: x, y
    !$omp declare variant (f1) match (construct={simd(uniform(y))})
  end
end subroutine
!CHECK: !$OMP DECLARE VARIANT (f1) MATCH(CONSTRUCT={SIMD(UNIFORM(y))})
!PARSE-TREE: | | | | DeclarationConstruct -> SpecificationConstruct -> OpenMPDeclarativeConstruct -> OmpDeclareVariantDirective
!PARSE-TREE-NEXT: | | | | | Verbatim
!PARSE-TREE-NEXT: | | | | | Name = 'f1'
!PARSE-TREE-NEXT: | | | | | OmpClauseList -> OmpClause -> Match -> OmpMatchClause -> OmpContextSelectorSpecification -> OmpTraitSetSelector
!PARSE-TREE-NEXT: | | | | | | OmpTraitSetSelectorName -> Value = Construct
!PARSE-TREE-NEXT: | | | | | | OmpTraitSelector
!PARSE-TREE-NEXT: | | | | | | | OmpTraitSelectorName -> Value = Simd
!PARSE-TREE-NEXT: | | | | | | | Properties
!PARSE-TREE-NEXT: | | | | | | | | OmpTraitProperty -> OmpClause -> Uniform -> Name = 'y'
