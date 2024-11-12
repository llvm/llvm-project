!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=52 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=52 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine f00(x)
  integer :: x
  !$omp target map(ompx_hold, always, present, close, to: x)
  x = x + 1
  !$omp end target
end

!UNPARSE: SUBROUTINE f00 (x)
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP TARGET  MAP(OMPX_HOLD, ALWAYS, PRESENT, CLOSE, TO: x)
!UNPARSE:   x=x+1_4
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: | OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | TypeModifier = Ompx_Hold
!PARSE-TREE: | | TypeModifier = Always
!PARSE-TREE: | | TypeModifier = Present
!PARSE-TREE: | | TypeModifier = Close
!PARSE-TREE: | | Type = To
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | bool = 'true'

subroutine f01(x)
  integer :: x
  !$omp target map(ompx_hold, always, present, close: x)
  x = x + 1
  !$omp end target
end

!UNPARSE: SUBROUTINE f01 (x)
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP TARGET  MAP(OMPX_HOLD, ALWAYS, PRESENT, CLOSE: x)
!UNPARSE:   x=x+1_4
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: | OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | TypeModifier = Ompx_Hold
!PARSE-TREE: | | TypeModifier = Always
!PARSE-TREE: | | TypeModifier = Present
!PARSE-TREE: | | TypeModifier = Close
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | bool = 'true'

subroutine f02(x)
  integer :: x
  !$omp target map(from: x)
  x = x + 1
  !$omp end target
end

!UNPARSE: SUBROUTINE f02 (x)
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP TARGET  MAP(FROM: x)
!UNPARSE:   x=x+1_4
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: | OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | Type = From
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | bool = 'true'

subroutine f03(x)
  integer :: x
  !$omp target map(x)
  x = x + 1
  !$omp end target
end

!UNPARSE: SUBROUTINE f03 (x)
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP TARGET  MAP(x)
!UNPARSE:   x=x+1_4
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: | OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | bool = 'true'

subroutine f04(x)
  integer :: x
  !$omp target map(ompx_hold always, present, close, to: x)
  x = x + 1
  !$omp end target
end

!UNPARSE: SUBROUTINE f04 (x)
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP TARGET  MAP(OMPX_HOLD, ALWAYS, PRESENT, CLOSE, TO: x)
!UNPARSE:   x=x+1_4
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: | OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | TypeModifier = Ompx_Hold
!PARSE-TREE: | | TypeModifier = Always
!PARSE-TREE: | | TypeModifier = Present
!PARSE-TREE: | | TypeModifier = Close
!PARSE-TREE: | | Type = To
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | bool = 'false'

subroutine f05(x)
  integer :: x
  !$omp target map(ompx_hold, always, present, close: x)
  x = x + 1
  !$omp end target
end

!UNPARSE: SUBROUTINE f05 (x)
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP TARGET  MAP(OMPX_HOLD, ALWAYS, PRESENT, CLOSE: x)
!UNPARSE:   x=x+1_4
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: | OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | TypeModifier = Ompx_Hold
!PARSE-TREE: | | TypeModifier = Always
!PARSE-TREE: | | TypeModifier = Present
!PARSE-TREE: | | TypeModifier = Close
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'

!PARSE-TREE: | | bool = 'true'

subroutine f10(x)
  integer :: x(10)
  !$omp target map(present, iterator(integer :: i = 1:10), to: x(i))
  x = x + 1
  !$omp end target
end

!UNPARSE: SUBROUTINE f10 (x)
!UNPARSE:  INTEGER x(10_4)
!UNPARSE: !$OMP TARGET  MAP(PRESENT, ITERATOR(INTEGER i = 1_4:10_4), TO: x(i))
!UNPARSE:   x=x+1_4
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: | OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | TypeModifier = Present
!PARSE-TREE: | | OmpIteratorModifier -> OmpIteratorSpecifier
!PARSE-TREE: | | | TypeDeclarationStmt
!PARSE-TREE: | | | | DeclarationTypeSpec -> IntrinsicTypeSpec -> IntegerTypeSpec ->
!PARSE-TREE: | | | | EntityDecl
!PARSE-TREE: | | | | | Name = 'i'
!PARSE-TREE: | | | SubscriptTriplet
!PARSE-TREE: | | | | Scalar -> Integer -> Expr = '1_4'
!PARSE-TREE: | | | | | LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | | | | Scalar -> Integer -> Expr = '10_4'
!PARSE-TREE: | | | | | LiteralConstant -> IntLiteralConstant = '10'
!PARSE-TREE: | | Type = To
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> ArrayElement
!PARSE-TREE: | | | DataRef -> Name = 'x'
!PARSE-TREE: | | | SectionSubscript -> Integer -> Expr = 'i'
!PARSE-TREE: | | | | Designator -> DataRef -> Name = 'i'
!PARSE-TREE: | | bool = 'true'

subroutine f11(x)
  integer :: x(10)
  !$omp target map(present, iterator(i = 1:10), to: x(i))
  x = x + 1
  !$omp end target
end

!UNPARSE: SUBROUTINE f11 (x)
!UNPARSE:  INTEGER x(10_4)
!UNPARSE: !$OMP TARGET  MAP(PRESENT, ITERATOR(INTEGER i = 1_4:10_4), TO: x(i))
!UNPARSE:   x=x+1_4
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: | OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | TypeModifier = Present
!PARSE-TREE: | | OmpIteratorModifier -> OmpIteratorSpecifier
!PARSE-TREE: | | | TypeDeclarationStmt
!PARSE-TREE: | | | | DeclarationTypeSpec -> IntrinsicTypeSpec -> IntegerTypeSpec ->
!PARSE-TREE: | | | | EntityDecl
!PARSE-TREE: | | | | | Name = 'i'
!PARSE-TREE: | | | SubscriptTriplet
!PARSE-TREE: | | | | Scalar -> Integer -> Expr = '1_4'
!PARSE-TREE: | | | | | LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | | | | Scalar -> Integer -> Expr = '10_4'
!PARSE-TREE: | | | | | LiteralConstant -> IntLiteralConstant = '10'
!PARSE-TREE: | | Type = To
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> ArrayElement
!PARSE-TREE: | | | DataRef -> Name = 'x'
!PARSE-TREE: | | | SectionSubscript -> Integer -> Expr = 'i'
!PARSE-TREE: | | | | Designator -> DataRef -> Name = 'i'
!PARSE-TREE: | | bool = 'true'

subroutine f12(x)
  integer :: x(10)
  !$omp target map(present, iterator(i = 1:10, integer :: j = 1:10), to: x((i + j) / 2))
  x = x + 1
  !$omp end target
end

!UNPARSE: SUBROUTINE f12 (x)
!UNPARSE:  INTEGER x(10_4)
!UNPARSE: !$OMP TARGET  MAP(PRESENT, ITERATOR(INTEGER i = 1_4:10_4, INTEGER j = 1_4:10_4), TO: x((i+j)/2_4))
!UNPARSE:   x=x+1_4
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: | OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | TypeModifier = Present
!PARSE-TREE: | | OmpIteratorModifier -> OmpIteratorSpecifier
!PARSE-TREE: | | | TypeDeclarationStmt
!PARSE-TREE: | | | | DeclarationTypeSpec -> IntrinsicTypeSpec -> IntegerTypeSpec ->
!PARSE-TREE: | | | | EntityDecl
!PARSE-TREE: | | | | | Name = 'i'
!PARSE-TREE: | | | SubscriptTriplet
!PARSE-TREE: | | | | Scalar -> Integer -> Expr = '1_4'
!PARSE-TREE: | | | | | LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | | | | Scalar -> Integer -> Expr = '10_4'
!PARSE-TREE: | | | | | LiteralConstant -> IntLiteralConstant = '10'
!PARSE-TREE: | | OmpIteratorSpecifier
!PARSE-TREE: | | | TypeDeclarationStmt
!PARSE-TREE: | | | | DeclarationTypeSpec -> IntrinsicTypeSpec -> IntegerTypeSpec ->
!PARSE-TREE: | | | | EntityDecl
!PARSE-TREE: | | | | | Name = 'j'
!PARSE-TREE: | | | SubscriptTriplet
!PARSE-TREE: | | | | Scalar -> Integer -> Expr = '1_4'
!PARSE-TREE: | | | | | LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | | | | Scalar -> Integer -> Expr = '10_4'
!PARSE-TREE: | | | | | LiteralConstant -> IntLiteralConstant = '10'
!PARSE-TREE: | | Type = To
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> ArrayElement
!PARSE-TREE: | | | DataRef -> Name = 'x'
!PARSE-TREE: | | | SectionSubscript -> Integer -> Expr = '(i+j)/2_4'
!PARSE-TREE: | | | | Divide
!PARSE-TREE: | | | | | Expr = '(i+j)'
!PARSE-TREE: | | | | | | Parentheses -> Expr = 'i+j'
!PARSE-TREE: | | | | | | | Add
!PARSE-TREE: | | | | | | | | Expr = 'i'
!PARSE-TREE: | | | | | | | | | Designator -> DataRef -> Name = 'i'
!PARSE-TREE: | | | | | | | | Expr = 'j'
!PARSE-TREE: | | | | | | | | | Designator -> DataRef -> Name = 'j'
!PARSE-TREE: | | | | | Expr = '2_4'
!PARSE-TREE: | | | | | | LiteralConstant -> IntLiteralConstant = '2'
!PARSE-TREE: | | bool = 'true'

subroutine f90(x, y)
  integer :: x(10)
  integer :: y
  integer, parameter :: p = 23
  !$omp target map(present, iterator(i, j = y:p, k = i:j), to: x(k))
  x = x + 1
  !$omp end target
end

!UNPARSE: SUBROUTINE f90 (x, y)
!UNPARSE:  INTEGER x(10_4)
!UNPARSE:  INTEGER y
!UNPARSE:  INTEGER, PARAMETER :: p = 23_4
!UNPARSE: !$OMP TARGET  MAP(PRESENT, ITERATOR(INTEGER i, j = y:23_4, INTEGER k = i:j), TO: x(k))
!UNPARSE:   x=x+1_4
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: | OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | TypeModifier = Present
!PARSE-TREE: | | OmpIteratorModifier -> OmpIteratorSpecifier
!PARSE-TREE: | | | TypeDeclarationStmt
!PARSE-TREE: | | | | DeclarationTypeSpec -> IntrinsicTypeSpec -> IntegerTypeSpec ->
!PARSE-TREE: | | | | EntityDecl
!PARSE-TREE: | | | | | Name = 'i'
!PARSE-TREE: | | | | EntityDecl
!PARSE-TREE: | | | | | Name = 'j'
!PARSE-TREE: | | | SubscriptTriplet
!PARSE-TREE: | | | | Scalar -> Integer -> Expr = 'y'
!PARSE-TREE: | | | | | Designator -> DataRef -> Name = 'y'
!PARSE-TREE: | | | | Scalar -> Integer -> Expr = '23_4'
!PARSE-TREE: | | | | | Designator -> DataRef -> Name = 'p'
!PARSE-TREE: | | OmpIteratorSpecifier
!PARSE-TREE: | | | TypeDeclarationStmt
!PARSE-TREE: | | | | DeclarationTypeSpec -> IntrinsicTypeSpec -> IntegerTypeSpec ->
!PARSE-TREE: | | | | EntityDecl
!PARSE-TREE: | | | | | Name = 'k'
!PARSE-TREE: | | | SubscriptTriplet
!PARSE-TREE: | | | | Scalar -> Integer -> Expr = 'i'
!PARSE-TREE: | | | | | Designator -> DataRef -> Name = 'i'
!PARSE-TREE: | | | | Scalar -> Integer -> Expr = 'j'
!PARSE-TREE: | | | | | Designator -> DataRef -> Name = 'j'
!PARSE-TREE: | | Type = To
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> ArrayElement
!PARSE-TREE: | | | DataRef -> Name = 'x'
!PARSE-TREE: | | | SectionSubscript -> Integer -> Expr = 'k'
!PARSE-TREE: | | | | Designator -> DataRef -> Name = 'k'
!PARSE-TREE: | | bool = 'true'

