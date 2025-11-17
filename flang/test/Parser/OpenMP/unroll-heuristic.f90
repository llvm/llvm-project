! RUN: %flang_fc1 -fopenmp -fopenmp-version=51 %s -fdebug-unparse         | FileCheck --check-prefix=UNPARSE %s
! RUN: %flang_fc1 -fopenmp -fopenmp-version=51 %s -fdebug-dump-parse-tree | FileCheck --check-prefix=PTREE %s

subroutine openmp_parse_unroll_heuristic
  integer i

  !$omp unroll
  do i = 1, 100
    call func(i)
  end do
  !$omp end unroll
END subroutine openmp_parse_unroll_heuristic


!UNPARSE:      !$OMP UNROLL
!UNPARSE-NEXT: DO i=1_4,100_4
!UNPARSE-NEXT:   CALL func(i)
!UNPARSE-NEXT: END DO
!UNPARSE-NEXT: !$OMP END UNROLL

!PTREE:      OpenMPConstruct -> OpenMPLoopConstruct
!PTREE-NEXT: | OmpBeginLoopDirective
!PTREE-NEXT: | | OmpDirectiveName -> llvm::omp::Directive = unroll
!PTREE-NEXT: | | OmpClauseList ->
!PTREE-NEXT: | | Flags = None
!PTREE-NEXT: | Block
!PTREE-NEXT: | | ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct
!PTREE-NEXT: | | | NonLabelDoStmt
!PTREE-NEXT: | | | | LoopControl -> LoopBounds
!PTREE-NEXT: | | | | | Scalar -> Name = 'i'
!PTREE-NEXT: | | | | | Scalar -> Expr = '1_4'
!PTREE-NEXT: | | | | | | LiteralConstant -> IntLiteralConstant = '1'
!PTREE-NEXT: | | | | | Scalar -> Expr = '100_4'
!PTREE-NEXT: | | | | | | LiteralConstant -> IntLiteralConstant = '100'
!PTREE-NEXT: | | | Block
!PTREE-NEXT: | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> CallStmt = 'CALL func(i)'
!PTREE-NEXT: | | | | | | | Call
!PTREE-NEXT: | | | | | | ProcedureDesignator -> Name = 'func'
!PTREE-NEXT: | | | | | | ActualArgSpec
!PTREE-NEXT: | | | | | | | ActualArg -> Expr = 'i'
!PTREE-NEXT: | | | | | | | | Designator -> DataRef -> Name = 'i'
!PTREE-NEXT: | | | EndDoStmt ->
!PTREE-NEXT: | OmpEndLoopDirective
!PTREE-NEXT: | | OmpDirectiveName -> llvm::omp::Directive = unroll
!PTREE-NEXT: | | OmpClauseList ->
!PTREE-NEXT: | | Flags = None
