! RUN: %flang_fc1 -fdebug-unparse -fopenmp %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp %s | FileCheck --check-prefix="PARSE-TREE" %s
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
!PARSE-TREE:        OmpReductionInitializerClause -> Expr = '0_4'
