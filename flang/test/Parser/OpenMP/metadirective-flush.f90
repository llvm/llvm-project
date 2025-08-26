!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=52 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=52 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine f00()
  integer :: x
  !$omp metadirective when(user={condition(.true.)}: flush seq_cst (x))
end

!UNPARSE: SUBROUTINE f00
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP METADIRECTIVE  WHEN(USER={CONDITION(.true._4)}: FLUSH SEQ_CST(x))
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpMetadirectiveDirective
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
!PARSE-TREE: | | | OmpDirectiveName -> llvm::omp::Directive = flush
!PARSE-TREE: | | | OmpArgumentList -> OmpArgument -> OmpLocator -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | OmpClauseList -> OmpClause -> SeqCst
!PARSE-TREE: | | | Flags = DeprecatedSyntax

subroutine f01()
  integer :: x
  !$omp metadirective when(user={condition(.true.)}: flush(x) seq_cst)
end

!UNPARSE: SUBROUTINE f01
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP METADIRECTIVE  WHEN(USER={CONDITION(.true._4)}: FLUSH(x) SEQ_CST)
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpMetadirectiveDirective
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
!PARSE-TREE: | | | OmpDirectiveName -> llvm::omp::Directive = flush
!PARSE-TREE: | | | OmpArgumentList -> OmpArgument -> OmpLocator -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | OmpClauseList -> OmpClause -> SeqCst
!PARSE-TREE: | | | Flags = None
