! RUN: %flang_fc1 -fdebug-unparse -fopenmp %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp %s | FileCheck --check-prefix="PARSE-TREE" %s

!CHECK-LABEL: SUBROUTINE initme (x, n)
subroutine initme(x,n)
  integer x,n
  x=n
end subroutine initme
!CHECK: END SUBROUTINE initme

!CHECK: FUNCTION func(x, n, init)
function func(x, n, init)
  integer func
  integer x(n)
  integer res
  interface
     subroutine initme(x,n)
       integer x,n
     end subroutine initme
  end interface
!$omp declare reduction(red_add:integer(4):omp_out=omp_out+omp_in) initializer(initme(omp_priv,0))
!CHECK: !$OMP DECLARE REDUCTION(red_add:INTEGER(KIND=4_4): omp_out=omp_out+omp_in) INITIA&
!CHECKL !$OMP&LIZER(initme(omp_priv,0))

!PARSE-TREE: DeclarationConstruct -> SpecificationConstruct -> OpenMPDeclarativeConstruct -> OpenMPDeclareReductionConstruct -> OmpDirectiveSpecification
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = declare reduction
!PARSE-TREE: | OmpArgumentList -> OmpArgument -> OmpReductionSpecifier
!PARSE-TREE: | | OmpReductionIdentifier -> ProcedureDesignator -> Name = 'red_add'
!PARSE-TREE: | | OmpTypeNameList -> OmpTypeName -> DeclarationTypeSpec -> IntrinsicTypeSpec -> IntegerTypeSpec -> KindSelector -> Scalar -> Integer -> Constant -> Expr = '4_4'
!PARSE-TREE: | | | LiteralConstant -> IntLiteralConstant = '4'
!PARSE-TREE: | | OmpCombinerExpression -> OmpStylizedInstance
!PARSE-TREE: | | | OmpStylizedDeclaration
!PARSE-TREE: | | | OmpStylizedDeclaration
!PARSE-TREE: | | | Instance -> AssignmentStmt = 'omp_out=omp_out+omp_in'
!PARSE-TREE: | | | | Variable = 'omp_out'
!PARSE-TREE: | | | | | Designator -> DataRef -> Name = 'omp_out'
!PARSE-TREE: | | | | Expr = 'omp_out+omp_in'
!PARSE-TREE: | | | | | Add
!PARSE-TREE: | | | | | | Expr = 'omp_out'
!PARSE-TREE: | | | | | | | Designator -> DataRef -> Name = 'omp_out'
!PARSE-TREE: | | | | | | Expr = 'omp_in'
!PARSE-TREE: | | | | | | | Designator -> DataRef -> Name = 'omp_in'
!PARSE-TREE: | OmpClauseList -> OmpClause -> Initializer -> OmpInitializerClause -> OmpInitializerExpression -> OmpStylizedInstance
!PARSE-TREE: | | OmpStylizedDeclaration
!PARSE-TREE: | | OmpStylizedDeclaration
!PARSE-TREE: | | Instance -> CallStmt = 'CALL initme(omp_priv,0_4)'
!PARSE-TREE: | | | Call
!PARSE-TREE: | | | | ProcedureDesignator -> Name = 'initme'
!PARSE-TREE: | | | | ActualArgSpec
!PARSE-TREE: | | | | | ActualArg -> Expr = 'omp_priv'
!PARSE-TREE: | | | | | | Designator -> DataRef -> Name = 'omp_priv'
!PARSE-TREE: | | | | ActualArgSpec
!PARSE-TREE: | | | | | ActualArg -> Expr = '0_4'
!PARSE-TREE: | | | | | | LiteralConstant -> IntLiteralConstant = '0'
!PARSE-TREE: | Flags = None

  res=init
!$omp simd reduction(red_add:res)
!CHECK: !$OMP SIMD REDUCTION(red_add: res)

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'res=init'
!PARSE-TREE: | Variable = 'res'
!PARSE-TREE: | | Designator -> DataRef -> Name = 'res'
!PARSE-TREE: | Expr = 'init'
!PARSE-TREE: | | Designator -> DataRef -> Name = 'init'
!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE: | OmpBeginLoopDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = simd
!PARSE-TREE: | | OmpClauseList -> OmpClause -> Reduction -> OmpReductionClause
!PARSE-TREE: | | | Modifier -> OmpReductionIdentifier -> ProcedureDesignator -> Name = 'red_add'
!PARSE-TREE: | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'res'
!PARSE-TREE: | | Flags = None
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct

  do i=1,n
     res=res+x(i)
  enddo
  func=res
end function func
!CHECK: END FUNCTION func

!CHECK-LABEL: program main
program main
  integer :: my_var
!CHECK: !$OMP DECLARE REDUCTION(my_add_red:INTEGER: omp_out = omp_out + omp_in) INITIA&
!CHECK: !$OMP&LIZER(omp_priv=0)

  !$omp declare reduction (my_add_red : integer : omp_out = omp_out + omp_in) initializer (omp_priv=0)
  my_var = 0
  !$omp parallel reduction (my_add_red : my_var) num_threads(4)
  my_var = omp_get_thread_num() + 1
  !$omp end parallel
  print *, "sum of thread numbers is ", my_var
end program main

!PARSE-TREE: DeclarationConstruct -> SpecificationConstruct -> OpenMPDeclarativeConstruct -> OpenMPDeclareReductionConstruct -> OmpDirectiveSpecification
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = declare reduction
!PARSE-TREE: | OmpArgumentList -> OmpArgument -> OmpReductionSpecifier
!PARSE-TREE: | | OmpReductionIdentifier -> ProcedureDesignator -> Name = 'my_add_red'
!PARSE-TREE: | | OmpTypeNameList -> OmpTypeName -> DeclarationTypeSpec -> IntrinsicTypeSpec -> IntegerTypeSpec ->
!PARSE-TREE: | | OmpCombinerExpression -> OmpStylizedInstance
!PARSE-TREE: | | | OmpStylizedDeclaration
!PARSE-TREE: | | | OmpStylizedDeclaration
!PARSE-TREE: | | | Instance -> AssignmentStmt = 'omp_out=omp_out+omp_in'
!PARSE-TREE: | | | | Variable = 'omp_out'
!PARSE-TREE: | | | | | Designator -> DataRef -> Name = 'omp_out'
!PARSE-TREE: | | | | Expr = 'omp_out+omp_in'
!PARSE-TREE: | | | | | Add
!PARSE-TREE: | | | | | | Expr = 'omp_out'
!PARSE-TREE: | | | | | | | Designator -> DataRef -> Name = 'omp_out'
!PARSE-TREE: | | | | | | Expr = 'omp_in'
!PARSE-TREE: | | | | | | | Designator -> DataRef -> Name = 'omp_in'
!PARSE-TREE: | OmpClauseList -> OmpClause -> Initializer -> OmpInitializerClause -> OmpInitializerExpression -> OmpStylizedInstance
!PARSE-TREE: | | OmpStylizedDeclaration
!PARSE-TREE: | | OmpStylizedDeclaration
!PARSE-TREE: | | Instance -> AssignmentStmt = 'omp_priv=0_4'
!PARSE-TREE: | | | Variable = 'omp_priv'
!PARSE-TREE: | | | | Designator -> DataRef -> Name = 'omp_priv'
!PARSE-TREE: | | | Expr = '0_4'
!PARSE-TREE: | | | | LiteralConstant -> IntLiteralConstant = '0'
!PARSE-TREE: | Flags = None
