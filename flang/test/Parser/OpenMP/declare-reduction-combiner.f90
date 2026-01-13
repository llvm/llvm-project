!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=60 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=60 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine f00
  type t
    integer :: x
  end type

  !$omp declare reduction(tred : t : omp_out%x = omp_out%x + omp_in%x)
end

!UNPARSE: SUBROUTINE f00
!UNPARSE:  TYPE :: t
!UNPARSE:   INTEGER :: x
!UNPARSE:  END TYPE
!UNPARSE: !$OMP DECLARE_REDUCTION(tred:t: omp_out%x = omp_out%x + omp_in%x)
!UNPARSE: END SUBROUTINE

!PARSE-TREE: DeclarationConstruct -> SpecificationConstruct -> OpenMPDeclarativeConstruct
!PARSE-TREE: OpenMPDeclareReductionConstruct -> OmpDirectiveSpecification
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = declare reduction
!PARSE-TREE: | OmpArgumentList -> OmpArgument -> OmpReductionSpecifier
!PARSE-TREE: | | OmpReductionIdentifier -> ProcedureDesignator -> Name = 'tred'
!PARSE-TREE: | | OmpTypeNameList -> OmpTypeName -> TypeSpec -> DerivedTypeSpec
!PARSE-TREE: | | | Name = 't'
!PARSE-TREE: | | OmpCombinerExpression -> OmpStylizedInstance
!PARSE-TREE: | | | OmpStylizedDeclaration
!PARSE-TREE: | | | OmpStylizedDeclaration
!PARSE-TREE: | | | Instance -> AssignmentStmt = 'omp_out%x=omp_out%x+omp_in%x'
!PARSE-TREE: | | | | Variable = 'omp_out%x'
!PARSE-TREE: | | | | | Designator -> DataRef -> StructureComponent
!PARSE-TREE: | | | | | | DataRef -> Name = 'omp_out'
!PARSE-TREE: | | | | | | Name = 'x'
!PARSE-TREE: | | | | Expr = 'omp_out%x+omp_in%x'
!PARSE-TREE: | | | | | Add
!PARSE-TREE: | | | | | | Expr = 'omp_out%x'
!PARSE-TREE: | | | | | | | Designator -> DataRef -> StructureComponent
!PARSE-TREE: | | | | | | | | DataRef -> Name = 'omp_out'
!PARSE-TREE: | | | | | | | | Name = 'x'
!PARSE-TREE: | | | | | | Expr = 'omp_in%x'
!PARSE-TREE: | | | | | | | Designator -> DataRef -> StructureComponent
!PARSE-TREE: | | | | | | | | DataRef -> Name = 'omp_in'
!PARSE-TREE: | | | | | | | | Name = 'x'
!PARSE-TREE: | OmpClauseList ->
!PARSE-TREE: | Flags = {}


subroutine f01
  type t
    integer :: x
  end type

  !$omp declare reduction(tred : t) combiner(omp_out%x = omp_out%x + omp_in%x)
end

!UNPARSE: SUBROUTINE f01
!UNPARSE:  TYPE :: t
!UNPARSE:   INTEGER :: x
!UNPARSE:  END TYPE
!UNPARSE: !$OMP DECLARE_REDUCTION(tred:t) COMBINER(omp_out%x = omp_out%x + omp_in%x)
!UNPARSE: END SUBROUTINE


!PARSE-TREE: DeclarationConstruct -> SpecificationConstruct -> OpenMPDeclarativeConstruct -> OpenMPDeclareReductionConstruct -> OmpDirectiveSpecification
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = declare reduction
!PARSE-TREE: | OmpArgumentList -> OmpArgument -> OmpReductionSpecifier
!PARSE-TREE: | | OmpReductionIdentifier -> ProcedureDesignator -> Name = 'tred'
!PARSE-TREE: | | OmpTypeNameList -> OmpTypeName -> TypeSpec -> DerivedTypeSpec
!PARSE-TREE: | | | Name = 't'
!PARSE-TREE: | OmpClauseList -> OmpClause -> Combiner -> OmpCombinerClause -> OmpCombinerExpression -> OmpStylizedInstance
!PARSE-TREE: | | OmpStylizedDeclaration
!PARSE-TREE: | | OmpStylizedDeclaration
!PARSE-TREE: | | Instance -> AssignmentStmt = 'omp_out%x=omp_out%x+omp_in%x'
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
!PARSE-TREE: | Flags = {}
