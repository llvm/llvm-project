!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=52 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=52 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine f00(x)
  integer :: x
  !$omp target update from(x)
end

!UNPARSE: SUBROUTINE f00 (x)
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP TARGET UPDATE  FROM(x)
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpSimpleStandaloneDirective -> llvm::omp::Directive = target update
!PARSE-TREE: OmpClauseList -> OmpClause -> From -> OmpFromClause
!PARSE-TREE: | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | bool = 'true'

subroutine f01(x)
  integer :: x
  !$omp target update from(present: x)
end

!UNPARSE: SUBROUTINE f01 (x)
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP TARGET UPDATE  FROM(PRESENT: x)
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpSimpleStandaloneDirective -> llvm::omp::Directive = target update
!PARSE-TREE: OmpClauseList -> OmpClause -> From -> OmpFromClause
!PARSE-TREE: | Modifier -> OmpExpectation -> Value = Present
!PARSE-TREE: | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | bool = 'true'

subroutine f02(x)
  integer :: x(10)
  !$omp target update from(present iterator(i = 1:10): x(i))
end

!UNPARSE: SUBROUTINE f02 (x)
!UNPARSE:  INTEGER x(10_4)
!UNPARSE: !$OMP TARGET UPDATE  FROM(PRESENT, ITERATOR(INTEGER i = 1_4:10_4): x(i))
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpSimpleStandaloneDirective -> llvm::omp::Directive = target update
!PARSE-TREE: OmpClauseList -> OmpClause -> From -> OmpFromClause
!PARSE-TREE: | Modifier -> OmpExpectation -> Value = Present
!PARSE-TREE: | Modifier -> OmpIterator -> OmpIteratorSpecifier
!PARSE-TREE: | | TypeDeclarationStmt
!PARSE-TREE: | | | DeclarationTypeSpec -> IntrinsicTypeSpec -> IntegerTypeSpec ->
!PARSE-TREE: | | | EntityDecl
!PARSE-TREE: | | | | Name = 'i'
!PARSE-TREE: | | SubscriptTriplet
!PARSE-TREE: | | | Scalar -> Integer -> Expr = '1_4'
!PARSE-TREE: | | | | LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | | | Scalar -> Integer -> Expr = '10_4'
!PARSE-TREE: | | | | LiteralConstant -> IntLiteralConstant = '10'
!PARSE-TREE: | OmpObjectList -> OmpObject -> Designator -> DataRef -> ArrayElement
!PARSE-TREE: | | DataRef -> Name = 'x'
!PARSE-TREE: | | SectionSubscript -> Integer -> Expr = 'i'
!PARSE-TREE: | | | Designator -> DataRef -> Name = 'i'
!PARSE-TREE: | bool = 'false'

subroutine f03(x)
  integer :: x(10)
  !$omp target update from(present, iterator(i = 1:10): x(i))
end

!UNPARSE: SUBROUTINE f03 (x)
!UNPARSE:  INTEGER x(10_4)
!UNPARSE: !$OMP TARGET UPDATE  FROM(PRESENT, ITERATOR(INTEGER i = 1_4:10_4): x(i))
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpSimpleStandaloneDirective -> llvm::omp::Directive = target update
!PARSE-TREE: OmpClauseList -> OmpClause -> From -> OmpFromClause
!PARSE-TREE: | Modifier -> OmpExpectation -> Value = Present
!PARSE-TREE: | Modifier -> OmpIterator -> OmpIteratorSpecifier
!PARSE-TREE: | | TypeDeclarationStmt
!PARSE-TREE: | | | DeclarationTypeSpec -> IntrinsicTypeSpec -> IntegerTypeSpec ->
!PARSE-TREE: | | | EntityDecl
!PARSE-TREE: | | | | Name = 'i'
!PARSE-TREE: | | SubscriptTriplet
!PARSE-TREE: | | | Scalar -> Integer -> Expr = '1_4'
!PARSE-TREE: | | | | LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | | | Scalar -> Integer -> Expr = '10_4'
!PARSE-TREE: | | | | LiteralConstant -> IntLiteralConstant = '10'
!PARSE-TREE: | OmpObjectList -> OmpObject -> Designator -> DataRef -> ArrayElement
!PARSE-TREE: | | DataRef -> Name = 'x'
!PARSE-TREE: | | SectionSubscript -> Integer -> Expr = 'i'
!PARSE-TREE: | | | Designator -> DataRef -> Name = 'i'
!PARSE-TREE: | bool = 'true'
