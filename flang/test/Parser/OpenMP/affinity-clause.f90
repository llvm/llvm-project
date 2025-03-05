!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=50 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=50 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine f00(x)
  integer :: x(10)
!$omp task affinity(x)
  x = x + 1
!$omp end task
end

!UNPARSE: SUBROUTINE f00 (x)
!UNPARSE:  INTEGER x(10_4)
!UNPARSE: !$OMP TASK  AFFINITY(x)
!UNPARSE:   x=x+1_4
!UNPARSE: !$OMP END TASK
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: | OmpBlockDirective -> llvm::omp::Directive = task
!PARSE-TREE: | OmpClauseList -> OmpClause -> Affinity -> OmpAffinityClause
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'

subroutine f01(x)
  integer :: x(10)
!$omp task affinity(x(1), x(3))
  x = x + 1
!$omp end task
end

!UNPARSE: SUBROUTINE f01 (x)
!UNPARSE:  INTEGER x(10_4)
!UNPARSE: !$OMP TASK  AFFINITY(x(1_4),x(3_4))
!UNPARSE:   x=x+1_4
!UNPARSE: !$OMP END TASK
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: | OmpBlockDirective -> llvm::omp::Directive = task
!PARSE-TREE: | OmpClauseList -> OmpClause -> Affinity -> OmpAffinityClause
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> ArrayElement
!PARSE-TREE: | | | DataRef -> Name = 'x'
!PARSE-TREE: | | | SectionSubscript -> Integer -> Expr = '1_4'
!PARSE-TREE: | | | | LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | | OmpObject -> Designator -> DataRef -> ArrayElement
!PARSE-TREE: | | | DataRef -> Name = 'x'
!PARSE-TREE: | | | SectionSubscript -> Integer -> Expr = '3_4'
!PARSE-TREE: | | | | LiteralConstant -> IntLiteralConstant = '3'

subroutine f02(x)
  integer :: x(10)
!$omp task affinity(iterator(i = 1:3): x(i))
  x = x + 1
!$omp end task
end

!UNPARSE: SUBROUTINE f02 (x)
!UNPARSE:  INTEGER x(10_4)
!UNPARSE: !$OMP TASK  AFFINITY(ITERATOR(INTEGER i = 1_4:3_4): x(i))
!UNPARSE:   x=x+1_4
!UNPARSE: !$OMP END TASK
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: | OmpBlockDirective -> llvm::omp::Directive = task
!PARSE-TREE: | OmpClauseList -> OmpClause -> Affinity -> OmpAffinityClause
!PARSE-TREE: | | Modifier -> OmpIterator -> OmpIteratorSpecifier
!PARSE-TREE: | | | TypeDeclarationStmt
!PARSE-TREE: | | | | DeclarationTypeSpec -> IntrinsicTypeSpec -> IntegerTypeSpec ->
!PARSE-TREE: | | | | EntityDecl
!PARSE-TREE: | | | | | Name = 'i'
!PARSE-TREE: | | | SubscriptTriplet
!PARSE-TREE: | | | | Scalar -> Integer -> Expr = '1_4'
!PARSE-TREE: | | | | | LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | | | | Scalar -> Integer -> Expr = '3_4'
!PARSE-TREE: | | | | | LiteralConstant -> IntLiteralConstant = '3'
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> ArrayElement
!PARSE-TREE: | | | DataRef -> Name = 'x'
!PARSE-TREE: | | | SectionSubscript -> Integer -> Expr = 'i'
!PARSE-TREE: | | | | Designator -> DataRef -> Name = 'i'
