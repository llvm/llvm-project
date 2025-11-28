! RUN: %flang_fc1 -fdebug-unparse -fopenmp %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp %s | FileCheck --check-prefix="PARSE-TREE" %s

!! Test multiple declarations for the same type, with different operations.
module mymod
  type :: tt
     real r
  end type tt
contains
  function mymax(a, b)
    type(tt) :: a, b, mymax
    if (a%r > b%r) then
       mymax = a
    else
       mymax = b
    end if
  end function mymax
end module mymod

program omp_examples
!CHECK-LABEL: PROGRAM omp_examples
  use mymod
  implicit none
  integer, parameter :: n = 100
  integer :: i
  type(tt) :: values(n), sum, prod, big, small

  !$omp declare reduction(+:tt:omp_out%r = omp_out%r + omp_in%r) initializer(omp_priv%r = 0)
!CHECK: !$OMP DECLARE REDUCTION(+:tt: omp_out%r = omp_out%r + omp_in%r) INITIALIZER(om&
!CHECK-NEXT: !$OMP&p_priv%r = 0)

!PARSE-TREE: DeclarationConstruct -> SpecificationConstruct -> OpenMPDeclarativeConstruct -> OpenMPDeclareReductionConstruct -> OmpDirectiveSpecification
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = declare reduction
!PARSE-TREE: | OmpArgumentList -> OmpArgument -> OmpReductionSpecifier
!PARSE-TREE: | | OmpReductionIdentifier -> DefinedOperator -> IntrinsicOperator = Add
!PARSE-TREE: | | OmpTypeNameList -> OmpTypeName -> TypeSpec -> DerivedTypeSpec
!PARSE-TREE: | | | Name = 'tt'
!PARSE-TREE: | | OmpCombinerExpression -> OmpStylizedInstance
!PARSE-TREE: | | | OmpStylizedDeclaration
!PARSE-TREE: | | | OmpStylizedDeclaration
!PARSE-TREE: | | | Instance -> AssignmentStmt = 'omp_out%r=omp_out%r+omp_in%r'
!PARSE-TREE: | | | | Variable = 'omp_out%r'
!PARSE-TREE: | | | | | Designator -> DataRef -> StructureComponent
!PARSE-TREE: | | | | | | DataRef -> Name = 'omp_out'
!PARSE-TREE: | | | | | | Name = 'r'
!PARSE-TREE: | | | | Expr = 'omp_out%r+omp_in%r'
!PARSE-TREE: | | | | | Add
!PARSE-TREE: | | | | | | Expr = 'omp_out%r'
!PARSE-TREE: | | | | | | | Designator -> DataRef -> StructureComponent
!PARSE-TREE: | | | | | | | | DataRef -> Name = 'omp_out'
!PARSE-TREE: | | | | | | | | Name = 'r'
!PARSE-TREE: | | | | | | Expr = 'omp_in%r'
!PARSE-TREE: | | | | | | | Designator -> DataRef -> StructureComponent
!PARSE-TREE: | | | | | | | | DataRef -> Name = 'omp_in'
!PARSE-TREE: | | | | | | | | Name = 'r'
!PARSE-TREE: | OmpClauseList -> OmpClause -> Initializer -> OmpInitializerClause -> OmpInitializerExpression -> OmpStylizedInstance
!PARSE-TREE: | | OmpStylizedDeclaration
!PARSE-TREE: | | OmpStylizedDeclaration
!PARSE-TREE: | | Instance -> AssignmentStmt = 'omp_priv%r=0._4'
!PARSE-TREE: | | | Variable = 'omp_priv%r'
!PARSE-TREE: | | | | Designator -> DataRef -> StructureComponent
!PARSE-TREE: | | | | | DataRef -> Name = 'omp_priv'
!PARSE-TREE: | | | | | Name = 'r'
!PARSE-TREE: | | | Expr = '0_4'
!PARSE-TREE: | | | | LiteralConstant -> IntLiteralConstant = '0'
!PARSE-TREE: | Flags = None

  !$omp declare reduction(*:tt:omp_out%r = omp_out%r * omp_in%r) initializer(omp_priv%r = 1)
!CHECK-NEXT: !$OMP DECLARE REDUCTION(*:tt: omp_out%r = omp_out%r * omp_in%r) INITIALIZER(om&
!CHECK-NEXT: !$OMP&p_priv%r = 1)

!PARSE-TREE: DeclarationConstruct -> SpecificationConstruct -> OpenMPDeclarativeConstruct -> OpenMPDeclareReductionConstruct -> OmpDirectiveSpecification
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = declare reduction
!PARSE-TREE: | OmpArgumentList -> OmpArgument -> OmpReductionSpecifier
!PARSE-TREE: | | OmpReductionIdentifier -> DefinedOperator -> IntrinsicOperator = Multiply
!PARSE-TREE: | | OmpTypeNameList -> OmpTypeName -> TypeSpec -> DerivedTypeSpec
!PARSE-TREE: | | | Name = 'tt'
!PARSE-TREE: | | OmpCombinerExpression -> OmpStylizedInstance
!PARSE-TREE: | | | OmpStylizedDeclaration
!PARSE-TREE: | | | OmpStylizedDeclaration
!PARSE-TREE: | | | Instance -> AssignmentStmt = 'omp_out%r=omp_out%r*omp_in%r'
!PARSE-TREE: | | | | Variable = 'omp_out%r'
!PARSE-TREE: | | | | | Designator -> DataRef -> StructureComponent
!PARSE-TREE: | | | | | | DataRef -> Name = 'omp_out'
!PARSE-TREE: | | | | | | Name = 'r'
!PARSE-TREE: | | | | Expr = 'omp_out%r*omp_in%r'
!PARSE-TREE: | | | | | Multiply
!PARSE-TREE: | | | | | | Expr = 'omp_out%r'
!PARSE-TREE: | | | | | | | Designator -> DataRef -> StructureComponent
!PARSE-TREE: | | | | | | | | DataRef -> Name = 'omp_out'
!PARSE-TREE: | | | | | | | | Name = 'r'
!PARSE-TREE: | | | | | | Expr = 'omp_in%r'
!PARSE-TREE: | | | | | | | Designator -> DataRef -> StructureComponent
!PARSE-TREE: | | | | | | | | DataRef -> Name = 'omp_in'
!PARSE-TREE: | | | | | | | | Name = 'r'
!PARSE-TREE: | OmpClauseList -> OmpClause -> Initializer -> OmpInitializerClause -> OmpInitializerExpression -> OmpStylizedInstance
!PARSE-TREE: | | OmpStylizedDeclaration
!PARSE-TREE: | | OmpStylizedDeclaration
!PARSE-TREE: | | Instance -> AssignmentStmt = 'omp_priv%r=1._4'
!PARSE-TREE: | | | Variable = 'omp_priv%r'
!PARSE-TREE: | | | | Designator -> DataRef -> StructureComponent
!PARSE-TREE: | | | | | DataRef -> Name = 'omp_priv'
!PARSE-TREE: | | | | | Name = 'r'
!PARSE-TREE: | | | Expr = '1_4'
!PARSE-TREE: | | | | LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | Flags = None

  !$omp declare reduction(max:tt:omp_out = mymax(omp_out, omp_in)) initializer(omp_priv%r = 0)
!CHECK-NEXT: !$OMP DECLARE REDUCTION(max:tt: omp_out = mymax(omp_out, omp_in)) INITIALIZER(&
!CHECK-NEXT: !$OMP&omp_priv%r = 0)

!PARSE-TREE: DeclarationConstruct -> SpecificationConstruct -> OpenMPDeclarativeConstruct -> OpenMPDeclareReductionConstruct -> OmpDirectiveSpecification
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = declare reduction
!PARSE-TREE: | OmpArgumentList -> OmpArgument -> OmpReductionSpecifier
!PARSE-TREE: | | OmpReductionIdentifier -> ProcedureDesignator -> Name = 'max'
!PARSE-TREE: | | OmpTypeNameList -> OmpTypeName -> TypeSpec -> DerivedTypeSpec
!PARSE-TREE: | | | Name = 'tt'
!PARSE-TREE: | | OmpCombinerExpression -> OmpStylizedInstance
!PARSE-TREE: | | | OmpStylizedDeclaration
!PARSE-TREE: | | | OmpStylizedDeclaration
!PARSE-TREE: | | | Instance -> AssignmentStmt = 'omp_out=mymax(omp_out,omp_in)'
!PARSE-TREE: | | | | Variable = 'omp_out'
!PARSE-TREE: | | | | | Designator -> DataRef -> Name = 'omp_out'
!PARSE-TREE: | | | | Expr = 'mymax(omp_out,omp_in)'
!PARSE-TREE: | | | | | FunctionReference -> Call
!PARSE-TREE: | | | | | | ProcedureDesignator -> Name = 'mymax'
!PARSE-TREE: | | | | | | ActualArgSpec
!PARSE-TREE: | | | | | | | ActualArg -> Expr = 'omp_out'
!PARSE-TREE: | | | | | | | | Designator -> DataRef -> Name = 'omp_out'
!PARSE-TREE: | | | | | | ActualArgSpec
!PARSE-TREE: | | | | | | | ActualArg -> Expr = 'omp_in'
!PARSE-TREE: | | | | | | | | Designator -> DataRef -> Name = 'omp_in'
!PARSE-TREE: | OmpClauseList -> OmpClause -> Initializer -> OmpInitializerClause -> OmpInitializerExpression -> OmpStylizedInstance
!PARSE-TREE: | | OmpStylizedDeclaration
!PARSE-TREE: | | OmpStylizedDeclaration
!PARSE-TREE: | | Instance -> AssignmentStmt = 'omp_priv%r=0._4'
!PARSE-TREE: | | | Variable = 'omp_priv%r'
!PARSE-TREE: | | | | Designator -> DataRef -> StructureComponent
!PARSE-TREE: | | | | | DataRef -> Name = 'omp_priv'
!PARSE-TREE: | | | | | Name = 'r'
!PARSE-TREE: | | | Expr = '0_4'
!PARSE-TREE: | | | | LiteralConstant -> IntLiteralConstant = '0'
!PARSE-TREE: | Flags = None

  !$omp declare reduction(min:tt:omp_out%r = min(omp_out%r, omp_in%r)) initializer(omp_priv%r = 1)
!CHECK-NEXT: !$OMP DECLARE REDUCTION(min:tt: omp_out%r = min(omp_out%r, omp_in%r)) INITIALI&
!CHECK-NEXT: !$OMP&ZER(omp_priv%r = 1)

!PARSE-TREE: DeclarationConstruct -> SpecificationConstruct -> OpenMPDeclarativeConstruct -> OpenMPDeclareReductionConstruct -> OmpDirectiveSpecification
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = declare reduction
!PARSE-TREE: | OmpArgumentList -> OmpArgument -> OmpReductionSpecifier
!PARSE-TREE: | | OmpReductionIdentifier -> ProcedureDesignator -> Name = 'min'
!PARSE-TREE: | | OmpTypeNameList -> OmpTypeName -> TypeSpec -> DerivedTypeSpec
!PARSE-TREE: | | | Name = 'tt'
!PARSE-TREE: | | OmpCombinerExpression -> OmpStylizedInstance
!PARSE-TREE: | | | OmpStylizedDeclaration
!PARSE-TREE: | | | OmpStylizedDeclaration
!PARSE-TREE: | | | Instance -> AssignmentStmt = 'omp_out%r=min(omp_out%r,omp_in%r)'
!PARSE-TREE: | | | | Variable = 'omp_out%r'
!PARSE-TREE: | | | | | Designator -> DataRef -> StructureComponent
!PARSE-TREE: | | | | | | DataRef -> Name = 'omp_out'
!PARSE-TREE: | | | | | | Name = 'r'
!PARSE-TREE: | | | | Expr = 'min(omp_out%r,omp_in%r)'
!PARSE-TREE: | | | | | FunctionReference -> Call
!PARSE-TREE: | | | | | | ProcedureDesignator -> Name = 'min'
!PARSE-TREE: | | | | | | ActualArgSpec
!PARSE-TREE: | | | | | | | ActualArg -> Expr = 'omp_out%r'
!PARSE-TREE: | | | | | | | | Designator -> DataRef -> StructureComponent
!PARSE-TREE: | | | | | | | | | DataRef -> Name = 'omp_out'
!PARSE-TREE: | | | | | | | | | Name = 'r'
!PARSE-TREE: | | | | | | ActualArgSpec
!PARSE-TREE: | | | | | | | ActualArg -> Expr = 'omp_in%r'
!PARSE-TREE: | | | | | | | | Designator -> DataRef -> StructureComponent
!PARSE-TREE: | | | | | | | | | DataRef -> Name = 'omp_in'
!PARSE-TREE: | | | | | | | | | Name = 'r'
!PARSE-TREE: | OmpClauseList -> OmpClause -> Initializer -> OmpInitializerClause -> OmpInitializerExpression -> OmpStylizedInstance
!PARSE-TREE: | | OmpStylizedDeclaration
!PARSE-TREE: | | OmpStylizedDeclaration
!PARSE-TREE: | | Instance -> AssignmentStmt = 'omp_priv%r=1._4'
!PARSE-TREE: | | | Variable = 'omp_priv%r'
!PARSE-TREE: | | | | Designator -> DataRef -> StructureComponent
!PARSE-TREE: | | | | | DataRef -> Name = 'omp_priv'
!PARSE-TREE: | | | | | Name = 'r'
!PARSE-TREE: | | | Expr = '1_4'
!PARSE-TREE: | | | | LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | Flags = None

  call random_number(values%r)

  sum%r = 0
  !$omp parallel do reduction(+:sum)
!CHECK: !$OMP PARALLEL DO REDUCTION(+: sum)

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE: | OmpBeginLoopDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = parallel do
!PARSE-TREE: | | OmpClauseList -> OmpClause -> Reduction -> OmpReductionClause
!PARSE-TREE: | | | Modifier -> OmpReductionIdentifier -> DefinedOperator -> IntrinsicOperator = Add
!PARSE-TREE: | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'sum'
!PARSE-TREE: | | Flags = None
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct

  do i = 1, n
     sum%r = sum%r + values(i)%r
  end do

  prod%r = 1
  !$omp parallel do reduction(*:prod)
!CHECK: !$OMP PARALLEL DO REDUCTION(*: prod)

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE: | OmpBeginLoopDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = parallel do
!PARSE-TREE: | | OmpClauseList -> OmpClause -> Reduction -> OmpReductionClause
!PARSE-TREE: | | | Modifier -> OmpReductionIdentifier -> DefinedOperator -> IntrinsicOperator = Multiply
!PARSE-TREE: | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'prod'
!PARSE-TREE: | | Flags = None
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct

  do i = 1, n
     prod%r = prod%r * (values(i)%r+0.6)
  end do

  big%r = 0
  !$omp parallel do reduction(max:big)
!CHECK:  $OMP PARALLEL DO REDUCTION(max: big)

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE: | OmpBeginLoopDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = parallel do
!PARSE-TREE: | | OmpClauseList -> OmpClause -> Reduction -> OmpReductionClause
!PARSE-TREE: | | | Modifier -> OmpReductionIdentifier -> ProcedureDesignator -> Name = 'max'
!PARSE-TREE: | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'big'
!PARSE-TREE: | | Flags = None
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct

  do i = 1, n
     big = mymax(values(i), big)
  end do

  small%r = 1
  !$omp parallel do reduction(min:small)
!CHECK: !$OMP PARALLEL DO REDUCTION(min: small)

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE: | OmpBeginLoopDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = parallel do
!PARSE-TREE: | | OmpClauseList -> OmpClause -> Reduction -> OmpReductionClause
!PARSE-TREE: | | | Modifier -> OmpReductionIdentifier -> ProcedureDesignator -> Name = 'min'
!PARSE-TREE: | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'small'
!PARSE-TREE: | | Flags = None
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct

  do i = 1, n
     small%r = min(values(i)%r, small%r)
  end do

  print *, values%r
  print *, "sum=", sum%r
  print *, "prod=", prod%r
  print *, "small=", small%r, " big=", big%r
end program omp_examples
