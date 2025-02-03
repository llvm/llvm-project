!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=52 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=52 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine f00
  !$omp metadirective when(construct={target, parallel}: nothing)
end

!UNPARSE: SUBROUTINE f00
!UNPARSE: !$OMP METADIRECTIVE  WHEN(CONSTRUCT={TARGET, PARALLEL}: NOTHING)
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OmpMetadirectiveDirective
!PARSE-TREE: | OmpClauseList -> OmpClause -> When -> OmpWhenClause
!PARSE-TREE: | | Modifier -> OmpContextSelectorSpecification -> OmpTraitSetSelector
!PARSE-TREE: | | | OmpTraitSetSelectorName -> Value = Construct
!PARSE-TREE: | | | OmpTraitSelector
!PARSE-TREE: | | | | OmpTraitSelectorName -> llvm::omp::Directive = target
!PARSE-TREE: | | | OmpTraitSelector
!PARSE-TREE: | | | | OmpTraitSelectorName -> llvm::omp::Directive = parallel
!PARSE-TREE: | | OmpDirectiveSpecification
!PARSE-TREE: | | | llvm::omp::Directive = nothing
!PARSE-TREE: | | | OmpClauseList ->

subroutine f01
  !$omp metadirective when(device={kind(host), device_num(1)}: nothing)
end

!UNPARSE: SUBROUTINE f01
!UNPARSE: !$OMP METADIRECTIVE  WHEN(DEVICE={KIND(host), DEVICE_NUM(1_4)}: NOTHING)
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OmpMetadirectiveDirective
!PARSE-TREE: | OmpClauseList -> OmpClause -> When -> OmpWhenClause
!PARSE-TREE: | | Modifier -> OmpContextSelectorSpecification -> OmpTraitSetSelector
!PARSE-TREE: | | | OmpTraitSetSelectorName -> Value = Device
!PARSE-TREE: | | | OmpTraitSelector
!PARSE-TREE: | | | | OmpTraitSelectorName -> Value = Kind
!PARSE-TREE: | | | | Properties
!PARSE-TREE: | | | | | OmpTraitProperty -> OmpTraitPropertyName -> string = 'host'
!PARSE-TREE: | | | OmpTraitSelector
!PARSE-TREE: | | | | OmpTraitSelectorName -> Value = Device_Num
!PARSE-TREE: | | | | Properties
!PARSE-TREE: | | | | | OmpTraitProperty -> Scalar -> Expr = '1_4'
!PARSE-TREE: | | | | | | LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | | OmpDirectiveSpecification
!PARSE-TREE: | | | llvm::omp::Directive = nothing
!PARSE-TREE: | | | OmpClauseList ->

subroutine f02
  !$omp metadirective when(target_device={kind(any), device_num(7)}: nothing)
end

!UNPARSE: SUBROUTINE f02
!UNPARSE: !$OMP METADIRECTIVE  WHEN(TARGET_DEVICE={KIND(any), DEVICE_NUM(7_4)}: NOTHING)
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OmpMetadirectiveDirective
!PARSE-TREE: | OmpClauseList -> OmpClause -> When -> OmpWhenClause
!PARSE-TREE: | | Modifier -> OmpContextSelectorSpecification -> OmpTraitSetSelector
!PARSE-TREE: | | | OmpTraitSetSelectorName -> Value = Target_Device
!PARSE-TREE: | | | OmpTraitSelector
!PARSE-TREE: | | | | OmpTraitSelectorName -> Value = Kind
!PARSE-TREE: | | | | Properties
!PARSE-TREE: | | | | | OmpTraitProperty -> OmpTraitPropertyName -> string = 'any'
!PARSE-TREE: | | | OmpTraitSelector
!PARSE-TREE: | | | | OmpTraitSelectorName -> Value = Device_Num
!PARSE-TREE: | | | | Properties
!PARSE-TREE: | | | | | OmpTraitProperty -> Scalar -> Expr = '7_4'
!PARSE-TREE: | | | | | | LiteralConstant -> IntLiteralConstant = '7'
!PARSE-TREE: | | OmpDirectiveSpecification
!PARSE-TREE: | | | llvm::omp::Directive = nothing
!PARSE-TREE: | | | OmpClauseList ->

subroutine f03
  !$omp metadirective &
  !$omp & when(implementation={atomic_default_mem_order(acq_rel)}: nothing)
end

!UNPARSE: SUBROUTINE f03
!UNPARSE: !$OMP METADIRECTIVE  WHEN(IMPLEMENTATION={ATOMIC_DEFAULT_MEM_ORDER(ACQ_REL)}: &
!UNPARSE: !$OMP&NOTHING)
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OmpMetadirectiveDirective
!PARSE-TREE: | OmpClauseList -> OmpClause -> When -> OmpWhenClause
!PARSE-TREE: | | Modifier -> OmpContextSelectorSpecification -> OmpTraitSetSelector
!PARSE-TREE: | | | OmpTraitSetSelectorName -> Value = Implementation
!PARSE-TREE: | | | OmpTraitSelector
!PARSE-TREE: | | | | OmpTraitSelectorName -> Value = Atomic_Default_Mem_Order
!PARSE-TREE: | | | | Properties
!PARSE-TREE: | | | | | OmpTraitProperty -> OmpClause -> AcqRel
!PARSE-TREE: | | OmpDirectiveSpecification
!PARSE-TREE: | | | llvm::omp::Directive = nothing
!PARSE-TREE: | | | OmpClauseList ->

subroutine f04
  !$omp metadirective &
  !$omp & when(implementation={extension(haha(1), foo(baz, "bar"(1)))}: nothing)
end

!UNPARSE: SUBROUTINE f04
!UNPARSE: !$OMP METADIRECTIVE  WHEN(IMPLEMENTATION={EXTENSION(haha(1_4), foo(baz,bar(1_4)))}: &
!UNPARSE: !$OMP&NOTHING)
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OmpMetadirectiveDirective
!PARSE-TREE: | OmpClauseList -> OmpClause -> When -> OmpWhenClause
!PARSE-TREE: | | Modifier -> OmpContextSelectorSpecification -> OmpTraitSetSelector
!PARSE-TREE: | | | OmpTraitSetSelectorName -> Value = Implementation
!PARSE-TREE: | | | OmpTraitSelector
!PARSE-TREE: | | | | OmpTraitSelectorName -> Value = Extension
!PARSE-TREE: | | | | Properties
!PARSE-TREE: | | | | | OmpTraitProperty -> OmpTraitPropertyExtension -> Complex
!PARSE-TREE: | | | | | | OmpTraitPropertyName -> string = 'haha'
!PARSE-TREE: | | | | | | OmpTraitPropertyExtension -> Scalar -> Expr = '1_4'
!PARSE-TREE: | | | | | | | LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | | | | | OmpTraitProperty -> OmpTraitPropertyExtension
!PARSE-TREE: | | | | | | OmpTraitPropertyName -> string = 'foo'
!PARSE-TREE: | | | | | | OmpTraitPropertyExtension -> OmpTraitPropertyName -> string = 'baz'
!PARSE-TREE: | | | | | | OmpTraitPropertyExtension -> Complex
!PARSE-TREE: | | | | | | | OmpTraitPropertyName -> string = 'bar'
!PARSE-TREE: | | | | | | | OmpTraitPropertyExtension -> Scalar -> Expr = '1_4'
!PARSE-TREE: | | | | | | | | LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | | OmpDirectiveSpecification
!PARSE-TREE: | | | llvm::omp::Directive = nothing
!PARSE-TREE: | | | OmpClauseList ->

subroutine f05(x)
  integer :: x
  !$omp metadirective &
  !$omp & when(user={condition(score(100): .true.)}: &
  !$omp &    parallel do reduction(+: x)) &
  !$omp & otherwise(nothing)
  do i = 1, 10
  enddo
end

!UNPARSE: SUBROUTINE f05 (x)
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP METADIRECTIVE  WHEN(USER={CONDITION(SCORE(100_4): .true._4)}: PARALLEL DO REDUCTION(+&
!UNPARSE: !$OMP&: x)) OTHERWISE(NOTHING)
!UNPARSE:  DO i=1_4,10_4
!UNPARSE:  END DO
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OmpMetadirectiveDirective
!PARSE-TREE: | OmpClauseList -> OmpClause -> When -> OmpWhenClause
!PARSE-TREE: | | Modifier -> OmpContextSelectorSpecification -> OmpTraitSetSelector
!PARSE-TREE: | | | OmpTraitSetSelectorName -> Value = User
!PARSE-TREE: | | | OmpTraitSelector
!PARSE-TREE: | | | | OmpTraitSelectorName -> Value = Condition
!PARSE-TREE: | | | | Properties
!PARSE-TREE: | | | | | OmpTraitScore -> Scalar -> Integer -> Expr = '100_4'
!PARSE-TREE: | | | | | | LiteralConstant -> IntLiteralConstant = '100'
!PARSE-TREE: | | | | | OmpTraitProperty -> Scalar -> Expr = '.true._4'
!PARSE-TREE: | | | | | | LiteralConstant -> LogicalLiteralConstant
!PARSE-TREE: | | | | | | | bool = 'true'
!PARSE-TREE: | | OmpDirectiveSpecification
!PARSE-TREE: | | | llvm::omp::Directive = parallel do
!PARSE-TREE: | | | OmpClauseList -> OmpClause -> Reduction -> OmpReductionClause
!PARSE-TREE: | | | | Modifier -> OmpReductionIdentifier -> DefinedOperator -> IntrinsicOperator = Add
!PARSE-TREE: | | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | OmpClause -> Otherwise -> OmpOtherwiseClause -> OmpDirectiveSpecification
!PARSE-TREE: | | llvm::omp::Directive = nothing
!PARSE-TREE: | | OmpClauseList ->

subroutine f06
  ! Two trait set selectors
  !$omp metadirective &
  !$omp & when(implementation={vendor("amd")}, &
  !$omp &      user={condition(.true.)}: nothing)
end

!UNPARSE: SUBROUTINE f06
!UNPARSE: !$OMP METADIRECTIVE  WHEN(IMPLEMENTATION={VENDOR(amd)}, USER={CONDITION(.true._4)}: NO&
!UNPARSE: !$OMP&THING)
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OmpMetadirectiveDirective
!PARSE-TREE: | OmpClauseList -> OmpClause -> When -> OmpWhenClause
!PARSE-TREE: | | Modifier -> OmpContextSelectorSpecification -> OmpTraitSetSelector
!PARSE-TREE: | | | OmpTraitSetSelectorName -> Value = Implementation
!PARSE-TREE: | | | OmpTraitSelector
!PARSE-TREE: | | | | OmpTraitSelectorName -> Value = Vendor
!PARSE-TREE: | | | | Properties
!PARSE-TREE: | | | | | OmpTraitProperty -> OmpTraitPropertyName -> string = 'amd'
!PARSE-TREE: | | OmpTraitSetSelector
!PARSE-TREE: | | | OmpTraitSetSelectorName -> Value = User
!PARSE-TREE: | | | OmpTraitSelector
!PARSE-TREE: | | | | OmpTraitSelectorName -> Value = Condition
!PARSE-TREE: | | | | Properties
!PARSE-TREE: | | | | | OmpTraitProperty -> Scalar -> Expr = '.true._4'
!PARSE-TREE: | | | | | | LiteralConstant -> LogicalLiteralConstant
!PARSE-TREE: | | | | | | | bool = 'true'
!PARSE-TREE: | | OmpDirectiveSpecification
!PARSE-TREE: | | | llvm::omp::Directive = nothing
!PARSE-TREE: | | | OmpClauseList ->

