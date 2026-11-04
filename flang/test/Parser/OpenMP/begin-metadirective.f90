!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=52 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=52 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine f00
  !$omp begin metadirective
  continue
  !$omp end metadirective
end

!UNPARSE: SUBROUTINE f00
!UNPARSE: !$OMP BEGIN METADIRECTIVE
!UNPARSE:  CONTINUE
!UNPARSE: !$OMP END METADIRECTIVE
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpDelimitedMetadirectiveDirective
!PARSE-TREE: | OmpBeginDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = metadirective
!PARSE-TREE: | | OmpClauseList ->
!PARSE-TREE: | | Flags = {ExplicitBegin}
!PARSE-TREE: | Block
!PARSE-TREE: | OmpEndDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = metadirective
!PARSE-TREE: | | OmpClauseList ->
!PARSE-TREE: | | Flags = {}

subroutine f01(s)
  integer :: i
  integer :: s
  s = 0
  !$omp begin metadirective &
  !$omp & when(user={condition(.true.)}: parallel do reduction(+: s)) &
  !$omp & otherwise(do)
  do i = 1, 10
    s = s + i
  end do
  !$omp end metadirective
end

!UNPARSE: SUBROUTINE f01 (s)
!UNPARSE:  INTEGER i
!UNPARSE:  INTEGER s
!UNPARSE:   s=0_4
!UNPARSE: !$OMP BEGIN METADIRECTIVE WHEN(USER={CONDITION(.true._4)}: PARALLEL DO REDUCTION(+: s)&
!UNPARSE: !$OMP&) OTHERWISE(DO)
!UNPARSE:  DO i=1_4,10_4
!UNPARSE:    s=s+i
!UNPARSE:  END DO
!UNPARSE: !$OMP END METADIRECTIVE
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpDelimitedMetadirectiveDirective
!PARSE-TREE: | OmpBeginDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = metadirective
!PARSE-TREE: | | OmpClauseList -> OmpClause -> When -> OmpWhenClause
!PARSE-TREE: | | | Modifier -> OmpContextSelectorSpecification -> OmpTraitSetSelector
!PARSE-TREE: | | | | OmpTraitSetSelectorName -> Value = User
!PARSE-TREE: | | | | OmpTraitSelector
!PARSE-TREE: | | | | | OmpTraitSelectorName -> Value = Condition
!PARSE-TREE: | | | | | Properties
!PARSE-TREE: | | | | | | OmpTraitProperty -> Scalar -> Expr = '.true._4'
!PARSE-TREE: | | | | | | | LiteralConstant -> LogicalLiteralConstant
!PARSE-TREE: | | | | | | | | bool = 'true'
!PARSE-TREE: | | | OmpDirectiveSpecification
!PARSE-TREE: | | | | OmpDirectiveName -> llvm::omp::Directive = parallel do
!PARSE-TREE: | | | | OmpClauseList -> OmpClause -> Reduction -> OmpReductionClause
!PARSE-TREE: | | | | | Modifier -> OmpReductionIdentifier -> DefinedOperator -> IntrinsicOperator = Add
!PARSE-TREE: | | | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 's'
!PARSE-TREE: | | | | Flags = {}
!PARSE-TREE: | | OmpClause -> Otherwise -> OmpOtherwiseClause -> OmpDirectiveSpecification
!PARSE-TREE: | | | OmpDirectiveName -> llvm::omp::Directive = do
!PARSE-TREE: | | | OmpClauseList ->
!PARSE-TREE: | | | Flags = {}
!PARSE-TREE: | | Flags = {ExplicitBegin}
!PARSE-TREE: | Block
!PARSE-TREE: | OmpEndDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = metadirective
!PARSE-TREE: | | OmpClauseList ->
!PARSE-TREE: | | Flags = {}

