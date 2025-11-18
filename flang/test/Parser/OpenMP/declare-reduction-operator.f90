! RUN: %flang_fc1 -fdebug-unparse -fopenmp %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp %s | FileCheck --check-prefix="PARSE-TREE" %s

!CHECK-LABEL: SUBROUTINE reduce_1 (n, tts)
subroutine reduce_1 ( n, tts )
  type :: tt
    integer :: x
    integer :: y
 end type tt
  type :: tt2
    real(8) :: x
    real(8) :: y
  end type
 
  integer :: n
  type(tt) :: tts(n)
  type(tt2) :: tts2(n)

!CHECK: !$OMP DECLARE REDUCTION(+:tt: omp_out = tt(omp_out%x - omp_in%x , omp_out%y - &
!CHECK: !$OMP&omp_in%y)) INITIALIZER(omp_priv = tt(0,0))

!PARSE-TREE: DeclarationConstruct -> SpecificationConstruct -> OpenMPDeclarativeConstruct -> OpenMPDeclareReductionConstruct -> OmpDirectiveSpecification
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = declare reduction
!PARSE-TREE: | OmpArgumentList -> OmpArgument -> OmpReductionSpecifier
!PARSE-TREE: | | OmpReductionIdentifier -> DefinedOperator -> IntrinsicOperator = Add
!PARSE-TREE: | | OmpTypeNameList -> OmpTypeName -> TypeSpec -> DerivedTypeSpec
!PARSE-TREE: | | | Name = 'tt'
!PARSE-TREE: | | OmpCombinerExpression -> OmpStylizedInstance
!PARSE-TREE: | | | OmpStylizedDeclaration
!PARSE-TREE: | | | OmpStylizedDeclaration
!PARSE-TREE: | | | Instance -> AssignmentStmt = 'omp_out=tt(x=omp_out%x-omp_in%x,y=omp_out%y-omp_in%y)'
!PARSE-TREE: | | | | Variable = 'omp_out'
!PARSE-TREE: | | | | | Designator -> DataRef -> Name = 'omp_out'
!PARSE-TREE: | | | | Expr = 'tt(x=omp_out%x-omp_in%x,y=omp_out%y-omp_in%y)'
!PARSE-TREE: | | | | | StructureConstructor
!PARSE-TREE: | | | | | | DerivedTypeSpec
!PARSE-TREE: | | | | | | | Name = 'tt'
!PARSE-TREE: | | | | | | ComponentSpec
!PARSE-TREE: | | | | | | | ComponentDataSource -> Expr = 'omp_out%x-omp_in%x'
!PARSE-TREE: | | | | | | | | Subtract
!PARSE-TREE: | | | | | | | | | Expr = 'omp_out%x'
!PARSE-TREE: | | | | | | | | | | Designator -> DataRef -> StructureComponent
!PARSE-TREE: | | | | | | | | | | | DataRef -> Name = 'omp_out'
!PARSE-TREE: | | | | | | | | | | | Name = 'x'
!PARSE-TREE: | | | | | | | | | Expr = 'omp_in%x'
!PARSE-TREE: | | | | | | | | | | Designator -> DataRef -> StructureComponent
!PARSE-TREE: | | | | | | | | | | | DataRef -> Name = 'omp_in'
!PARSE-TREE: | | | | | | | | | | | Name = 'x'
!PARSE-TREE: | | | | | | ComponentSpec
!PARSE-TREE: | | | | | | | ComponentDataSource -> Expr = 'omp_out%y-omp_in%y'
!PARSE-TREE: | | | | | | | | Subtract
!PARSE-TREE: | | | | | | | | | Expr = 'omp_out%y'
!PARSE-TREE: | | | | | | | | | | Designator -> DataRef -> StructureComponent
!PARSE-TREE: | | | | | | | | | | | DataRef -> Name = 'omp_out'
!PARSE-TREE: | | | | | | | | | | | Name = 'y'
!PARSE-TREE: | | | | | | | | | Expr = 'omp_in%y'
!PARSE-TREE: | | | | | | | | | | Designator -> DataRef -> StructureComponent
!PARSE-TREE: | | | | | | | | | | | DataRef -> Name = 'omp_in'
!PARSE-TREE: | | | | | | | | | | | Name = 'y'
!PARSE-TREE: | OmpClauseList -> OmpClause -> Initializer -> OmpInitializerClause -> OmpInitializerExpression -> OmpStylizedInstance
!PARSE-TREE: | | OmpStylizedDeclaration
!PARSE-TREE: | | OmpStylizedDeclaration
!PARSE-TREE: | | Instance -> AssignmentStmt = 'omp_priv=tt(x=0_4,y=0_4)'
!PARSE-TREE: | | | Variable = 'omp_priv'
!PARSE-TREE: | | | | Designator -> DataRef -> Name = 'omp_priv'
!PARSE-TREE: | | | Expr = 'tt(x=0_4,y=0_4)'
!PARSE-TREE: | | | | StructureConstructor
!PARSE-TREE: | | | | | DerivedTypeSpec
!PARSE-TREE: | | | | | | Name = 'tt'
!PARSE-TREE: | | | | | ComponentSpec
!PARSE-TREE: | | | | | | ComponentDataSource -> Expr = '0_4'
!PARSE-TREE: | | | | | | | LiteralConstant -> IntLiteralConstant = '0'
!PARSE-TREE: | | | | | ComponentSpec
!PARSE-TREE: | | | | | | ComponentDataSource -> Expr = '0_4'
!PARSE-TREE: | | | | | | | LiteralConstant -> IntLiteralConstant = '0'
!PARSE-TREE: | Flags = None
  !$omp declare reduction(+ : tt :  omp_out = tt(omp_out%x - omp_in%x , omp_out%y - omp_in%y)) initializer(omp_priv = tt(0,0))

  
!CHECK: !$OMP DECLARE REDUCTION(+:tt2: omp_out = tt2(omp_out%x - omp_in%x , omp_out%y &
!CHECK: !$OMP&- omp_in%y)) INITIALIZER(omp_priv = tt2(0,0))

!PARSE-TREE: DeclarationConstruct -> SpecificationConstruct -> OpenMPDeclarativeConstruct -> OpenMPDeclareReductionConstruct -> OmpDirectiveSpecification
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = declare reduction
!PARSE-TREE: | OmpArgumentList -> OmpArgument -> OmpReductionSpecifier
!PARSE-TREE: | | OmpReductionIdentifier -> DefinedOperator -> IntrinsicOperator = Add
!PARSE-TREE: | | OmpTypeNameList -> OmpTypeName -> TypeSpec -> DerivedTypeSpec
!PARSE-TREE: | | | Name = 'tt2'
!PARSE-TREE: | | OmpCombinerExpression -> OmpStylizedInstance
!PARSE-TREE: | | | OmpStylizedDeclaration
!PARSE-TREE: | | | OmpStylizedDeclaration
!PARSE-TREE: | | | Instance -> AssignmentStmt = 'omp_out=tt2(x=omp_out%x-omp_in%x,y=omp_out%y-omp_in%y)'
!PARSE-TREE: | | | | Variable = 'omp_out'
!PARSE-TREE: | | | | | Designator -> DataRef -> Name = 'omp_out'
!PARSE-TREE: | | | | Expr = 'tt2(x=omp_out%x-omp_in%x,y=omp_out%y-omp_in%y)'
!PARSE-TREE: | | | | | StructureConstructor
!PARSE-TREE: | | | | | | DerivedTypeSpec
!PARSE-TREE: | | | | | | | Name = 'tt2'
!PARSE-TREE: | | | | | | ComponentSpec
!PARSE-TREE: | | | | | | | ComponentDataSource -> Expr = 'omp_out%x-omp_in%x'
!PARSE-TREE: | | | | | | | | Subtract
!PARSE-TREE: | | | | | | | | | Expr = 'omp_out%x'
!PARSE-TREE: | | | | | | | | | | Designator -> DataRef -> StructureComponent
!PARSE-TREE: | | | | | | | | | | | DataRef -> Name = 'omp_out'
!PARSE-TREE: | | | | | | | | | | | Name = 'x'
!PARSE-TREE: | | | | | | | | | Expr = 'omp_in%x'
!PARSE-TREE: | | | | | | | | | | Designator -> DataRef -> StructureComponent
!PARSE-TREE: | | | | | | | | | | | DataRef -> Name = 'omp_in'
!PARSE-TREE: | | | | | | | | | | | Name = 'x'
!PARSE-TREE: | | | | | | ComponentSpec
!PARSE-TREE: | | | | | | | ComponentDataSource -> Expr = 'omp_out%y-omp_in%y'
!PARSE-TREE: | | | | | | | | Subtract
!PARSE-TREE: | | | | | | | | | Expr = 'omp_out%y'
!PARSE-TREE: | | | | | | | | | | Designator -> DataRef -> StructureComponent
!PARSE-TREE: | | | | | | | | | | | DataRef -> Name = 'omp_out'
!PARSE-TREE: | | | | | | | | | | | Name = 'y'
!PARSE-TREE: | | | | | | | | | Expr = 'omp_in%y'
!PARSE-TREE: | | | | | | | | | | Designator -> DataRef -> StructureComponent
!PARSE-TREE: | | | | | | | | | | | DataRef -> Name = 'omp_in'
!PARSE-TREE: | | | | | | | | | | | Name = 'y'
!PARSE-TREE: | OmpClauseList -> OmpClause -> Initializer -> OmpInitializerClause -> OmpInitializerExpression -> OmpStylizedInstance
!PARSE-TREE: | | OmpStylizedDeclaration
!PARSE-TREE: | | OmpStylizedDeclaration
!PARSE-TREE: | | Instance -> AssignmentStmt = 'omp_priv=tt2(x=0._8,y=0._8)'
!PARSE-TREE: | | | Variable = 'omp_priv'
!PARSE-TREE: | | | | Designator -> DataRef -> Name = 'omp_priv'
!PARSE-TREE: | | | Expr = 'tt2(x=0._8,y=0._8)'
!PARSE-TREE: | | | | StructureConstructor
!PARSE-TREE: | | | | | DerivedTypeSpec
!PARSE-TREE: | | | | | | Name = 'tt2'
!PARSE-TREE: | | | | | ComponentSpec
!PARSE-TREE: | | | | | | ComponentDataSource -> Expr = '0_4'
!PARSE-TREE: | | | | | | | LiteralConstant -> IntLiteralConstant = '0'
!PARSE-TREE: | | | | | ComponentSpec
!PARSE-TREE: | | | | | | ComponentDataSource -> Expr = '0_4'
!PARSE-TREE: | | | | | | | LiteralConstant -> IntLiteralConstant = '0'
!PARSE-TREE: | Flags = None
  !$omp declare reduction(+ :tt2 :  omp_out = tt2(omp_out%x - omp_in%x , omp_out%y - omp_in%y)) initializer(omp_priv = tt2(0,0))
  
  type(tt) :: diffp = tt( 0, 0 )
  type(tt2) :: diffp2 = tt2( 0, 0 )
  integer :: i

  !$omp parallel do reduction(+ : diffp)
  do i = 1, n
     diffp%x = diffp%x + tts(i)%x
     diffp%y = diffp%y + tts(i)%y
  end do

  !$omp parallel do reduction(+ : diffp2)
  do i = 1, n
     diffp2%x = diffp2%x + tts2(i)%x
     diffp2%y = diffp2%y + tts2(i)%y
  end do

end subroutine reduce_1
!CHECK: END SUBROUTINE reduce_1
