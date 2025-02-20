!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=50 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=50 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine f01
  continue
  !$omp metadirective &
  !$omp & when(user={condition(.true.)}: nothing) &
  !$omp & default(nothing)
end

!UNPARSE: SUBROUTINE f01
!UNPARSE:  CONTINUE
!UNPARSE: !$OMP METADIRECTIVE  WHEN(USER={CONDITION(.true._4)}: NOTHING) DEFAULT(NOTHING)
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OmpMetadirectiveDirective
!PARSE-TREE: | OmpClauseList -> OmpClause -> When -> OmpWhenClause
!PARSE-TREE: | | Modifier -> OmpContextSelectorSpecification -> OmpTraitSetSelector
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
!PARSE-TREE: | OmpClause -> Default -> OmpDefaultClause -> OmpDirectiveSpecification
!PARSE-TREE: | | llvm::omp::Directive = nothing
!PARSE-TREE: | | OmpClauseList ->
