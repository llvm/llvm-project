!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=60 %s -o - | FileCheck --check-prefix=UNPARSE %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=60 %s -o - | FileCheck --check-prefix=PARSE-TREE %s

! The directives to check:
!   cancellation_point
!   declare_mapper
!   declare_reduction
!   declare_simd
!   declare_target
!   declare_variant
!   target_data
!   target_enter_data
!   target_exit_data
!   target_update

subroutine f00
  implicit none
  integer :: i

  !$omp parallel
  do i = 1, 10
    !$omp cancellation_point parallel
  enddo
  !$omp end parallel
end

!UNPARSE: SUBROUTINE f00
!UNPARSE:  IMPLICIT NONE
!UNPARSE:  INTEGER i
!UNPARSE: !$OMP PARALLEL
!UNPARSE:  DO i=1_4,10_4
!UNPARSE: !$OMP CANCELLATION_POINT PARALLEL
!UNPARSE:  END DO
!UNPARSE: !$OMP END PARALLEL
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPCancellationPointConstruct -> OmpDirectiveSpecification
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = cancellation point
!PARSE-TREE: | OmpClauseList -> OmpClause -> CancellationConstructType -> OmpCancellationConstructTypeClause
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = parallel
!PARSE-TREE: | Flags = None

subroutine f01
  type :: t
    integer :: x
  end type
  !$omp declare_mapper(t :: v) map(v%x)
end

!UNPARSE: SUBROUTINE f01
!UNPARSE:  TYPE :: t
!UNPARSE:   INTEGER :: x
!UNPARSE:  END TYPE
!UNPARSE: !$OMP DECLARE MAPPER (t::v) MAP(v%x)
!UNPARSE: END SUBROUTINE

!PARSE-TREE: DeclarationConstruct -> SpecificationConstruct -> OpenMPDeclarativeConstruct -> OpenMPDeclareMapperConstruct
!PARSE-TREE: | Verbatim
!PARSE-TREE: | OmpMapperSpecifier
!PARSE-TREE: | | string = 't.omp.default.mapper'
!PARSE-TREE: | | TypeSpec -> DerivedTypeSpec
!PARSE-TREE: | | | Name = 't'
!PARSE-TREE: | | Name = 'v'
!PARSE-TREE: | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> StructureComponent
!PARSE-TREE: | | | DataRef -> Name = 'v'
!PARSE-TREE: | | | Name = 'x'
!PARSE-TREE: | | bool = 'true'

subroutine f02
  type :: t
    integer :: x
  end type
  !$omp declare_reduction(+ : t : omp_out%x = omp_out%x + omp_in%x)
end

!UNPARSE: SUBROUTINE f02
!UNPARSE:  TYPE :: t
!UNPARSE:   INTEGER :: x
!UNPARSE:  END TYPE
!UNPARSE: !$OMP DECLARE REDUCTION (+:t: omp_out%x=omp_out%x+omp_in%x
!UNPARSE: )
!UNPARSE: END SUBROUTINE

!PARSE-TREE: DeclarationConstruct -> SpecificationConstruct -> OpenMPDeclarativeConstruct -> OpenMPDeclareReductionConstruct
!PARSE-TREE: | Verbatim
!PARSE-TREE: | OmpReductionSpecifier
!PARSE-TREE: | | OmpReductionIdentifier -> DefinedOperator -> IntrinsicOperator = Add
!PARSE-TREE: | | OmpTypeNameList -> OmpTypeSpecifier -> TypeSpec -> DerivedTypeSpec
!PARSE-TREE: | | | Name = 't'
!PARSE-TREE: | | OmpReductionCombiner -> AssignmentStmt = 'omp_out%x=omp_out%x+omp_in%x'
!PARSE-TREE: | | | Variable = 'omp_out%x'
!PARSE-TREE: | | | | Designator -> DataRef -> StructureComponent
!PARSE-TREE: | | | | | DataRef -> Name = 'omp_out'
!PARSE-TREE: | | | | | Name = 'x'
!PARSE-TREE: | | | Expr = 'omp_out%x+omp_in%x'
!PARSE-TREE: | | | | Add
!PARSE-TREE: | | | | | Expr = 'omp_out%x'
!PARSE-TREE: | | | | | | Designator -> DataRef -> StructureComponent
!PARSE-TREE: | | | | | | | DataRef -> Name = 'omp_out'
!PARSE-TREE: | | | | | | | Name = 'x'
!PARSE-TREE: | | | | | Expr = 'omp_in%x'
!PARSE-TREE: | | | | | | Designator -> DataRef -> StructureComponent
!PARSE-TREE: | | | | | | | DataRef -> Name = 'omp_in'
!PARSE-TREE: | | | | | | | Name = 'x'
!PARSE-TREE: | OmpClauseList ->

subroutine f03
  !$omp declare_simd
end

!UNPARSE: SUBROUTINE f03
!UNPARSE: !$OMP DECLARE SIMD
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OpenMPDeclarativeConstruct -> OpenMPDeclareSimdConstruct
!PARSE-TREE: | Verbatim
!PARSE-TREE: | OmpClauseList ->

subroutine f04
  !$omp declare_target
end

!UNPARSE: SUBROUTINE f04
!UNPARSE: !$OMP DECLARE TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OpenMPDeclarativeConstruct -> OpenMPDeclareTargetConstruct
!PARSE-TREE: | Verbatim
!PARSE-TREE: | OmpDeclareTargetSpecifier -> OmpDeclareTargetWithClause -> OmpClauseList ->

subroutine f05
  implicit none
  interface
    subroutine g05
    end
  end interface
  !$omp declare_variant(g05) match(user={condition(.true.)})
end

!UNPARSE: SUBROUTINE f05
!UNPARSE:  IMPLICIT NONE
!UNPARSE:  INTERFACE
!UNPARSE:   SUBROUTINE g05
!UNPARSE:   END SUBROUTINE
!UNPARSE:  END INTERFACE
!UNPARSE: !$OMP DECLARE VARIANT (g05) MATCH(USER={CONDITION(.true._4)})
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OpenMPDeclarativeConstruct -> OmpDeclareVariantDirective
!PARSE-TREE: | Verbatim
!PARSE-TREE: | Name = 'g05'
!PARSE-TREE: | OmpClauseList -> OmpClause -> Match -> OmpMatchClause -> OmpContextSelectorSpecification -> OmpTraitSetSelector
!PARSE-TREE: | | OmpTraitSetSelectorName -> Value = User
!PARSE-TREE: | | OmpTraitSelector
!PARSE-TREE: | | | OmpTraitSelectorName -> Value = Condition
!PARSE-TREE: | | | Properties
!PARSE-TREE: | | | | OmpTraitProperty -> Scalar -> Expr = '.true._4'
!PARSE-TREE: | | | | | LiteralConstant -> LogicalLiteralConstant
!PARSE-TREE: | | | | | | bool = 'true'

subroutine f06
  implicit none
  integer :: i
  !$omp target_data map(tofrom: i)
  i = 0
  !$omp end target data
end

!UNPARSE: SUBROUTINE f06
!UNPARSE:  IMPLICIT NONE
!UNPARSE:  INTEGER i
!UNPARSE: !$OMP TARGET_DATA MAP(TOFROM: i)
!UNPARSE:   i=0_4
!UNPARSE: !$OMP END TARGET_DATA
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpBlockConstruct
!PARSE-TREE: | OmpBeginDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = target data
!PARSE-TREE: | | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | | Modifier -> OmpMapType -> Value = Tofrom
!PARSE-TREE: | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'i'
!PARSE-TREE: | | | bool = 'true'
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'i=0_4'
!PARSE-TREE: | | | Variable = 'i'
!PARSE-TREE: | | | | Designator -> DataRef -> Name = 'i'
!PARSE-TREE: | | | Expr = '0_4'
!PARSE-TREE: | | | | LiteralConstant -> IntLiteralConstant = '0'
!PARSE-TREE: | OmpEndDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = target data
!PARSE-TREE: | | OmpClauseList ->

subroutine f07
  implicit none
  integer :: i
  !$omp target_enter_data map(to: i)
end

!UNPARSE: SUBROUTINE f07
!UNPARSE:  IMPLICIT NONE
!UNPARSE:  INTEGER i
!UNPARSE: !$OMP TARGET_ENTER_DATA MAP(TO: i)
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPSimpleStandaloneConstruct -> OmpDirectiveSpecification
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = target enter data
!PARSE-TREE: | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | Modifier -> OmpMapType -> Value = To
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'i'
!PARSE-TREE: | | bool = 'true'
!PARSE-TREE: | Flags = None

subroutine f08
  implicit none
  integer :: i
  !$omp target_exit_data map(from: i)
end

!UNPARSE: SUBROUTINE f08
!UNPARSE:  IMPLICIT NONE
!UNPARSE:  INTEGER i
!UNPARSE: !$OMP TARGET_EXIT_DATA MAP(FROM: i)
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPSimpleStandaloneConstruct -> OmpDirectiveSpecification
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = target exit data
!PARSE-TREE: | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | Modifier -> OmpMapType -> Value = From
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'i'
!PARSE-TREE: | | bool = 'true'
!PARSE-TREE: | Flags = None

subroutine f09
  implicit none
  integer :: i
  !$omp target_update to(i)
end

!UNPARSE: SUBROUTINE f09
!UNPARSE:  IMPLICIT NONE
!UNPARSE:  INTEGER i
!UNPARSE: !$OMP TARGET_UPDATE TO(i)
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPSimpleStandaloneConstruct -> OmpDirectiveSpecification
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = target update
!PARSE-TREE: | OmpClauseList -> OmpClause -> To -> OmpToClause
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'i'
!PARSE-TREE: | | bool = 'true'
!PARSE-TREE: | Flags = None
