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
!CHECK: !$OMP DECLARE REDUCTION (red_add:INTEGER(KIND=4_4): omp_out=omp_out+omp_in
!CHECK: ) INITIALIZER(initme(omp_priv, 0_4))  
!$omp declare reduction(red_add:integer(4):omp_out=omp_out+omp_in) initializer(initme(omp_priv,0))
!PARSE-TREE:  DeclarationConstruct -> SpecificationConstruct -> OpenMPDeclarativeConstruct -> OpenMPDeclareReductionConstruct
!PARSE-TREE: OmpReductionCombiner -> AssignmentStmt = 'omp_out=omp_out+omp_in'
!PARSE-TREE:    OmpReductionInitializerClause -> OmpReductionInitializerProc
!PARSE-TREE-NEXT: ProcedureDesignator -> Name = 'initme'
  res=init
!$omp simd reduction(red_add:res)
!CHECK: !$OMP SIMD REDUCTION(red_add: res)
!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE:  OmpBeginLoopDirective
!PARSE-TREE:  OmpLoopDirective -> llvm::omp::Directive = simd
!PARSE-TREE:  OmpClauseList -> OmpClause -> Reduction -> OmpReductionClause
!PARSE-TREE:  Modifier -> OmpReductionIdentifier -> ProcedureDesignator -> Name = 'red_add
  do i=1,n
     res=res+x(i)
  enddo
  func=res
end function func
!CHECK: END FUNCTION func

!CHECK-LABEL: program main
program main
  integer :: my_var
!CHECK: !$OMP DECLARE REDUCTION (my_add_red:INTEGER: omp_out=omp_out+omp_in
!CHECK-NEXT: ) INITIALIZER(OMP_PRIV = 0_4)

  !$omp declare reduction (my_add_red : integer : omp_out = omp_out + omp_in) initializer (omp_priv=0)
  my_var = 0
  !$omp parallel reduction (my_add_red : my_var) num_threads(4)
  my_var = omp_get_thread_num() + 1
  !$omp end parallel
  print *, "sum of thread numbers is ", my_var
end program main

!PARSE-TREE:      OpenMPDeclareReductionConstruct
!PARSE-TREE:        OmpReductionIdentifier -> ProcedureDesignator -> Name = 'my_add_red'
!PARSE-TREE:        DeclarationTypeSpec -> IntrinsicTypeSpec -> IntegerTypeSpec
!PARSE-TREE:        OmpReductionCombiner -> AssignmentStmt = 'omp_out=omp_out+omp_in'
!PARSE-TREE:        OmpReductionInitializerClause -> OmpReductionInitializerExpr -> Expr = '0_4'
