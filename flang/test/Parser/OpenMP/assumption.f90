!RUN: split-file %s %t
!RUN: %flang_fc1 -fopenmp-version=51 -fopenmp -fdebug-unparse-no-sema %t/bare.f90 | FileCheck --check-prefixes=UNPARSE,UNPARSE5X %s
!RUN: %flang_fc1 -fopenmp-version=51 -fopenmp -fdebug-dump-parse-tree-no-sema %t/bare.f90 | FileCheck --check-prefix="PARSE-TREE" %s
!RUN: %flang_fc1 -fopenmp-version=60 -fopenmp -fdebug-unparse-no-sema %s | FileCheck --check-prefixes=UNPARSE,UNPARSE60,CONDITIONAL %s
!RUN: %flang_fc1 -fopenmp-version=60 -fopenmp -fdebug-dump-parse-tree-no-sema %s | FileCheck --check-prefixes=PARSE-TREE,CONDITIONAL-TREE %s

!--- bare.f90
subroutine sub1
  integer :: r
  !$omp assume no_openmp
  !$omp end assume

  !$omp assume no_parallelism
  !$omp end assume

  !$omp assume no_openmp_routines
  !$omp end assume

  !$omp assume absent(allocate), contains(workshare, task)
  block ! strictly-structured-block
  end block

  !$omp assume holds(1.eq.1)
  block
  end block
  print *, r
end subroutine sub1

!UNPARSE: SUBROUTINE sub1
!UNPARSE:  INTEGER r
!UNPARSE: !$OMP ASSUME NO_OPENMP
!UNPARSE: !$OMP END ASSUME
!UNPARSE: !$OMP ASSUME NO_PARALLELISM
!UNPARSE: !$OMP END ASSUME
!UNPARSE: !$OMP ASSUME NO_OPENMP_ROUTINES
!UNPARSE: !$OMP END ASSUME
!UNPARSE: !$OMP ASSUME ABSENT(ALLOCATE) CONTAINS(WORKSHARE,TASK)
!UNPARSE:  BLOCK
!UNPARSE:  END BLOCK
!UNPARSE: !$OMP ASSUME HOLDS(1==1)
!UNPARSE:  BLOCK
!UNPARSE:  END BLOCK
!UNPARSE:  PRINT *, r
!UNPARSE: END SUBROUTINE sub1

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpAssumeDirective
!PARSE-TREE: | OmpBeginDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = assume
!PARSE-TREE: | | OmpClauseList -> OmpClause -> NoOpenmp
!PARSE-TREE: | | Flags = {}
!PARSE-TREE: | Block
!PARSE-TREE: | OmpEndDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = assume
!PARSE-TREE: | | OmpClauseList ->
!PARSE-TREE: | | Flags = {}
!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpAssumeDirective
!PARSE-TREE: | OmpBeginDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = assume
!PARSE-TREE: | | OmpClauseList -> OmpClause -> NoParallelism
!PARSE-TREE: | | Flags = {}
!PARSE-TREE: | Block
!PARSE-TREE: | OmpEndDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = assume
!PARSE-TREE: | | OmpClauseList ->
!PARSE-TREE: | | Flags = {}
!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpAssumeDirective
!PARSE-TREE: | OmpBeginDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = assume
!PARSE-TREE: | | OmpClauseList -> OmpClause -> NoOpenmpRoutines
!PARSE-TREE: | | Flags = {}
!PARSE-TREE: | Block
!PARSE-TREE: | OmpEndDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = assume
!PARSE-TREE: | | OmpClauseList ->
!PARSE-TREE: | | Flags = {}
!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpAssumeDirective
!PARSE-TREE: | OmpBeginDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = assume
!PARSE-TREE: | | OmpClauseList -> OmpClause -> Absent -> OmpAbsentClause -> llvm::omp::Directive = allocate
!PARSE-TREE: | | OmpClause -> Contains -> OmpContainsClause -> llvm::omp::Directive = workshare
!PARSE-TREE: | | llvm::omp::Directive = task
!PARSE-TREE: | | Flags = {}
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> BlockConstruct
!PARSE-TREE: | | | BlockStmt ->
!PARSE-TREE: | | | BlockSpecificationPart -> SpecificationPart
!PARSE-TREE: | | | | ImplicitPart ->
!PARSE-TREE: | | | Block
!PARSE-TREE: | | | EndBlockStmt ->
!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpAssumeDirective
!PARSE-TREE: | OmpBeginDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = assume
!PARSE-TREE: | | OmpClauseList -> OmpClause -> Holds -> OmpHoldsClause -> Expr -> EQ
!PARSE-TREE: | | | Expr -> LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | | | Expr -> LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | | Flags = {}
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> BlockConstruct
!PARSE-TREE: | | | BlockStmt ->
!PARSE-TREE: | | | BlockSpecificationPart -> SpecificationPart
!PARSE-TREE: | | | | ImplicitPart ->
!PARSE-TREE: | | | Block
!PARSE-TREE: | | | EndBlockStmt ->


subroutine sub2
  integer :: r
  integer :: v
  v = 87
  !$omp assume no_openmp
  r = r + 1
  !$omp end assume
end subroutine sub2

!UNPARSE: SUBROUTINE sub2
!UNPARSE:  INTEGER r
!UNPARSE:  INTEGER v
!UNPARSE:  v = 87
!UNPARSE: !$OMP ASSUME NO_OPENMP
!UNPARSE:  r = r+1
!UNPARSE: !$OMP END ASSUME
!UNPARSE: END SUBROUTINE sub2

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt
!PARSE-TREE: | Variable -> Designator -> DataRef -> Name = 'v'
!PARSE-TREE: | Expr -> LiteralConstant -> IntLiteralConstant = '87'
!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpAssumeDirective
!PARSE-TREE: | OmpBeginDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = assume
!PARSE-TREE: | | OmpClauseList -> OmpClause -> NoOpenmp
!PARSE-TREE: | | Flags = {}
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt
!PARSE-TREE: | | | Variable -> Designator -> DataRef -> Name = 'r'
!PARSE-TREE: | | | Expr -> Add
!PARSE-TREE: | | | | Expr -> Designator -> DataRef -> Name = 'r'
!PARSE-TREE: | | | | Expr -> LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | OmpEndDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = assume
!PARSE-TREE: | | OmpClauseList ->
!PARSE-TREE: | | Flags = {}

program p
  !$omp assumes no_openmp
end program p

!UNPARSE: PROGRAM p
!UNPARSE: !$OMP ASSUMES NO_OPENMP
!UNPARSE: END PROGRAM p

!PARSE-TREE: OpenMPDeclarativeConstruct -> OmpAssumesDirective -> OmpDirectiveSpecification
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = assumes
!PARSE-TREE: | OmpClauseList -> OmpClause -> NoOpenmp
!PARSE-TREE: | Flags = {}
!PARSE-TREE: ImplicitPart ->

subroutine assumes_parallelism
  implicit none
  !$omp assumes no_parallelism
  !$omp assumes no_openmp no_parallelism
  !$omp assumes no_parallelism no_openmp
end subroutine assumes_parallelism

!UNPARSE: SUBROUTINE assumes_parallelism
!UNPARSE: IMPLICIT NONE
!UNPARSE: !$OMP ASSUMES NO_PARALLELISM
!UNPARSE: !$OMP ASSUMES NO_OPENMP NO_PARALLELISM
!UNPARSE: !$OMP ASSUMES NO_PARALLELISM NO_OPENMP
!UNPARSE: END SUBROUTINE assumes_parallelism

!PARSE-TREE: OpenMPDeclarativeConstruct -> OmpAssumesDirective -> OmpDirectiveSpecification
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = assumes
!PARSE-TREE: | OmpClauseList -> OmpClause -> NoParallelism
!PARSE-TREE: OpenMPDeclarativeConstruct -> OmpAssumesDirective -> OmpDirectiveSpecification
!PARSE-TREE: | OmpClauseList -> OmpClause -> NoOpenmp
!PARSE-TREE: | | OmpClause -> NoParallelism
!PARSE-TREE: OpenMPDeclarativeConstruct -> OmpAssumesDirective -> OmpDirectiveSpecification
!PARSE-TREE: | OmpClauseList -> OmpClause -> NoParallelism
!PARSE-TREE: | | OmpClause -> NoOpenmp

subroutine assumes_absent
  !$omp assumes absent(parallel, barrier)
  !$omp assumes absent(target enter data)
end subroutine assumes_absent

!UNPARSE: SUBROUTINE assumes_absent
!UNPARSE: !$OMP ASSUMES ABSENT(PARALLEL,BARRIER)
!UNPARSE5X: !$OMP ASSUMES ABSENT(TARGET ENTER DATA)
!UNPARSE60: !$OMP ASSUMES ABSENT(TARGET_ENTER_DATA)
!UNPARSE: END SUBROUTINE assumes_absent

!PARSE-TREE: OpenMPDeclarativeConstruct -> OmpAssumesDirective -> OmpDirectiveSpecification
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = assumes
!PARSE-TREE: | OmpClauseList -> OmpClause -> Absent -> OmpAbsentClause -> llvm::omp::Directive = parallel
!PARSE-TREE: | llvm::omp::Directive = barrier
!PARSE-TREE: | Flags = {}
!PARSE-TREE: OpenMPDeclarativeConstruct -> OmpAssumesDirective -> OmpDirectiveSpecification
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = assumes
!PARSE-TREE: | OmpClauseList -> OmpClause -> Absent -> OmpAbsentClause -> llvm::omp::Directive = target enter data
!PARSE-TREE: | Flags = {}

!--- conditional.f90
subroutine conditional_parallelism
  !$omp assumes no_parallelism(.true.)
  !$omp assumes no_parallelism(.false.)
end subroutine conditional_parallelism

!CONDITIONAL: SUBROUTINE conditional_parallelism
!CONDITIONAL: !$OMP ASSUMES NO_PARALLELISM(.TRUE.)
!CONDITIONAL: !$OMP ASSUMES NO_PARALLELISM(.FALSE.)
!CONDITIONAL: END SUBROUTINE conditional_parallelism

!CONDITIONAL-TREE: NoParallelism -> OmpNoParallelismClause
!CONDITIONAL-TREE: LogicalLiteralConstant
!CONDITIONAL-TREE: NoParallelism -> OmpNoParallelismClause
!CONDITIONAL-TREE: LogicalLiteralConstant
