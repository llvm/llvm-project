!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=52 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=52 %s | FileCheck --check-prefix="PARSE-TREE" %s

!Directive specification where directives have arguments

subroutine f00(x)
  integer :: x(10)
  !$omp metadirective when(user={condition(.true.)}: &
  !$omp & allocate(x))
end

!UNPARSE: SUBROUTINE f00 (x)
!UNPARSE:  INTEGER x(10_4)
!UNPARSE: !$OMP METADIRECTIVE  WHEN(USER={CONDITION(.true._4)}: ALLOCATE(x))
!UNPARSE: END SUBROUTINE

!PARSE-TREE: DeclarationConstruct -> SpecificationConstruct -> OpenMPDeclarativeConstruct -> OmpMetadirectiveDirective
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
!PARSE-TREE: | | | llvm::omp::Directive = allocate
!PARSE-TREE: | | | OmpArgument -> OmpLocator -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | OmpClauseList ->

subroutine f01(x)
  integer :: x
  !$omp metadirective when(user={condition(.true.)}: &
  !$omp & critical(x))
end

!UNPARSE: SUBROUTINE f01 (x)
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP METADIRECTIVE  WHEN(USER={CONDITION(.true._4)}: CRITICAL(x))
!UNPARSE: END SUBROUTINE

!PARSE-TREE: DeclarationConstruct -> SpecificationConstruct -> OpenMPDeclarativeConstruct -> OmpMetadirectiveDirective
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
!PARSE-TREE: | | | llvm::omp::Directive = critical
!PARSE-TREE: | | | OmpArgument -> OmpLocator -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | OmpClauseList ->

subroutine f02
  !$omp metadirective when(user={condition(.true.)}: &
  !$omp & declare mapper(mymapper : integer :: v) map(tofrom: v))
end

!UNPARSE: SUBROUTINE f02
!UNPARSE: !$OMP METADIRECTIVE  WHEN(USER={CONDITION(.true._4)}: DECLARE MAPPER(mymapper:INTEGER:&
!UNPARSE: !$OMP&:v) MAP(TOFROM: v))
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OpenMPDeclarativeConstruct -> OmpMetadirectiveDirective
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
!PARSE-TREE: | | | llvm::omp::Directive = declare mapper
!PARSE-TREE: | | | OmpArgument -> OmpMapperSpecifier
!PARSE-TREE: | | | | Name = 'mymapper'
!PARSE-TREE: | | | | TypeSpec -> IntrinsicTypeSpec -> IntegerTypeSpec ->
!PARSE-TREE: | | | | Name = 'v'
!PARSE-TREE: | | | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | | | Modifier -> OmpMapType -> Value = Tofrom
!PARSE-TREE: | | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'v'
!PARSE-TREE: | | | | bool = 'true'
!PARSE-TREE: ImplicitPart ->

subroutine f03
  type :: tt1
    integer :: x
  endtype
  type :: tt2
    real :: a
  endtype
  !$omp metadirective when(user={condition(.true.)}: &
  !$omp & declare reduction(+ : tt1, tt2 : omp_out = omp_in + omp_out))
end

!UNPARSE: SUBROUTINE f03
!UNPARSE:  TYPE :: tt1
!UNPARSE:   INTEGER :: x
!UNPARSE:  END TYPE
!UNPARSE:  TYPE :: tt2
!UNPARSE:   REAL :: a
!UNPARSE:  END TYPE
!UNPARSE: !$OMP METADIRECTIVE  WHEN(USER={CONDITION(.true._4)}: DECLARE REDUCTION(+:tt1,tt2: omp_out=omp_in+omp_out
!UNPARSE: ))
!UNPARSE: END SUBROUTINE

!PARSE-TREE: DeclarationConstruct -> SpecificationConstruct -> OpenMPDeclarativeConstruct -> OmpMetadirectiveDirective
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
!PARSE-TREE: | | | llvm::omp::Directive = declare reduction
!PARSE-TREE: | | | OmpArgument -> OmpReductionSpecifier
!PARSE-TREE: | | | | OmpReductionIdentifier -> DefinedOperator -> IntrinsicOperator = Add
!PARSE-TREE: | | | | OmpTypeNameList -> OmpTypeSpecifier -> TypeSpec -> DerivedTypeSpec
!PARSE-TREE: | | | | | Name = 'tt1'
!PARSE-TREE: | | | | OmpTypeSpecifier -> TypeSpec -> DerivedTypeSpec
!PARSE-TREE: | | | | | Name = 'tt2'
!PARSE-TREE: | | | | OmpReductionCombiner -> AssignmentStmt = 'omp_out=omp_in+omp_out'
!PARSE-TREE: | | | | | Variable = 'omp_out'
!PARSE-TREE: | | | | | | Designator -> DataRef -> Name = 'omp_out'
!PARSE-TREE: | | | | | Expr = 'omp_in+omp_out'
!PARSE-TREE: | | | | | | Add
!PARSE-TREE: | | | | | | | Expr = 'omp_in'
!PARSE-TREE: | | | | | | | | Designator -> DataRef -> Name = 'omp_in'
!PARSE-TREE: | | | | | | | Expr = 'omp_out'
!PARSE-TREE: | | | | | | | | Designator -> DataRef -> Name = 'omp_out'
!PARSE-TREE: | | | OmpClauseList ->

subroutine f04
  !$omp metadirective when(user={condition(.true.)}: &
  !$omp & declare simd(f04))
end

!UNPARSE: SUBROUTINE f04
!UNPARSE: !$OMP METADIRECTIVE  WHEN(USER={CONDITION(.true._4)}: DECLARE SIMD(f04))
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OpenMPDeclarativeConstruct -> OmpMetadirectiveDirective
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
!PARSE-TREE: | | | llvm::omp::Directive = declare simd
!PARSE-TREE: | | | OmpArgument -> OmpLocator -> OmpObject -> Designator -> DataRef -> Name = 'f04'
!PARSE-TREE: | | | OmpClauseList ->
!PARSE-TREE: ImplicitPart ->

subroutine f05
  !$omp metadirective when(user={condition(.true.)}: &
  !$omp & declare target(f05))
end

!UNPARSE: SUBROUTINE f05
!UNPARSE: !$OMP METADIRECTIVE  WHEN(USER={CONDITION(.true._4)}: DECLARE TARGET(f05))
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OpenMPDeclarativeConstruct -> OmpMetadirectiveDirective
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
!PARSE-TREE: | | | llvm::omp::Directive = declare target
!PARSE-TREE: | | | OmpArgument -> OmpLocator -> OmpObject -> Designator -> DataRef -> Name = 'f05'
!PARSE-TREE: | | | OmpClauseList ->
!PARSE-TREE: ImplicitPart ->

subroutine f06(x, y)
  integer :: x, y
  !$omp metadirective when(user={condition(.true.)}: &
  !$omp & flush(x, y))
end

!UNPARSE: SUBROUTINE f06 (x, y)
!UNPARSE:  INTEGER x, y
!UNPARSE: !$OMP METADIRECTIVE  WHEN(USER={CONDITION(.true._4)}: FLUSH(x, y))
!UNPARSE: END SUBROUTINE

!PARSE-TREE: DeclarationConstruct -> SpecificationConstruct -> OpenMPDeclarativeConstruct -> OmpMetadirectiveDirective
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
!PARSE-TREE: | | | llvm::omp::Directive = flush
!PARSE-TREE: | | | OmpArgument -> OmpLocator -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | OmpArgument -> OmpLocator -> OmpObject -> Designator -> DataRef -> Name = 'y'
!PARSE-TREE: | | | OmpClauseList ->

subroutine f07
  integer :: t
  !$omp metadirective when(user={condition(.true.)}: &
  !$omp & threadprivate(t))
end

!UNPARSE: SUBROUTINE f07
!UNPARSE:  INTEGER t
!UNPARSE: !$OMP METADIRECTIVE  WHEN(USER={CONDITION(.true._4)}: THREADPRIVATE(t))
!UNPARSE: END SUBROUTINE

!PARSE-TREE: DeclarationConstruct -> SpecificationConstruct -> OpenMPDeclarativeConstruct -> OmpMetadirectiveDirective
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
!PARSE-TREE: | | | llvm::omp::Directive = threadprivate
!PARSE-TREE: | | | OmpArgument -> OmpLocator -> OmpObject -> Designator -> DataRef -> Name = 't'
!PARSE-TREE: | | | OmpClauseList ->
