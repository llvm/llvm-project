!RUN: %flang_fc1 -fdebug-unparse -fopenmp %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp %s | FileCheck --check-prefix="PARSE-TREE" %s

! Check that this is parsed correctly. Specifically, that the "10 continue"
! terminates both do-loops, and both "parallel do" directives.

subroutine f
  !$omp parallel do lastprivate(i)
  do 10 i=1,2
  !$omp parallel do lastprivate(j)
    do 10 j=1,2
  10 continue
  !$omp parallel
  !$omp end parallel
  print *,'pass'
end

!UNPARSE: SUBROUTINE f
!UNPARSE: !$OMP PARALLEL DO LASTPRIVATE(i)
!UNPARSE:  DO i=1_4,2_4
!UNPARSE: !$OMP PARALLEL DO LASTPRIVATE(j)
!UNPARSE:   DO j=1_4,2_4
!UNPARSE:    10 CONTINUE
!UNPARSE:   END DO
!UNPARSE:  END DO
!UNPARSE: !$OMP PARALLEL
!UNPARSE: !$OMP END PARALLEL
!UNPARSE:  PRINT *, "pass"
!UNPARSE: END SUBROUTINE

!PARSE-TREE: Program -> ProgramUnit -> SubroutineSubprogram
!PARSE-TREE: | SubroutineStmt
!PARSE-TREE: | | Name = 'f'
!PARSE-TREE: | SpecificationPart
!PARSE-TREE: | | ImplicitPart ->
!PARSE-TREE: | ExecutionPart -> Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE: | | | OmpBeginDirective
!PARSE-TREE: | | | | OmpDirectiveName -> llvm::omp::Directive = parallel do
!PARSE-TREE: | | | | OmpClauseList -> OmpClause -> Lastprivate -> OmpLastprivateClause
!PARSE-TREE: | | | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'i'
!PARSE-TREE: | | | | Flags = {}
!PARSE-TREE: | | | Block
!PARSE-TREE: | | | | ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct
!PARSE-TREE: | | | | | NonLabelDoStmt
!PARSE-TREE: | | | | | | LoopControl -> LoopBounds
!PARSE-TREE: | | | | | | | Scalar -> Name = 'i'
!PARSE-TREE: | | | | | | | Scalar -> Expr = '1_4'
!PARSE-TREE: | | | | | | | | LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | | | | | | | Scalar -> Expr = '2_4'
!PARSE-TREE: | | | | | | | | LiteralConstant -> IntLiteralConstant = '2'
!PARSE-TREE: | | | | | Block
!PARSE-TREE: | | | | | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE: | | | | | | | OmpBeginDirective
!PARSE-TREE: | | | | | | | | OmpDirectiveName -> llvm::omp::Directive = parallel do
!PARSE-TREE: | | | | | | | | OmpClauseList -> OmpClause -> Lastprivate -> OmpLastprivateClause
!PARSE-TREE: | | | | | | | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'j'
!PARSE-TREE: | | | | | | | | Flags = {CrossesLabelDo}
!PARSE-TREE: | | | | | | | Block
!PARSE-TREE: | | | | | | | | ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct
!PARSE-TREE: | | | | | | | | | NonLabelDoStmt
!PARSE-TREE: | | | | | | | | | | LoopControl -> LoopBounds
!PARSE-TREE: | | | | | | | | | | | Scalar -> Name = 'j'
!PARSE-TREE: | | | | | | | | | | | Scalar -> Expr = '1_4'
!PARSE-TREE: | | | | | | | | | | | | LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | | | | | | | | | | | Scalar -> Expr = '2_4'
!PARSE-TREE: | | | | | | | | | | | | LiteralConstant -> IntLiteralConstant = '2'
!PARSE-TREE: | | | | | | | | | Block
!PARSE-TREE: | | | | | | | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> ContinueStmt
!PARSE-TREE: | | | | | | | | | EndDoStmt ->
!PARSE-TREE: | | | | | EndDoStmt ->
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpBlockConstruct
!PARSE-TREE: | | | OmpBeginDirective
!PARSE-TREE: | | | | OmpDirectiveName -> llvm::omp::Directive = parallel
!PARSE-TREE: | | | | OmpClauseList ->
!PARSE-TREE: | | | | Flags = {}
!PARSE-TREE: | | | Block
!PARSE-TREE: | | | OmpEndDirective
!PARSE-TREE: | | | | OmpDirectiveName -> llvm::omp::Directive = parallel
!PARSE-TREE: | | | | OmpClauseList ->
!PARSE-TREE: | | | | Flags = {}
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> PrintStmt
!PARSE-TREE: | | | Format -> Star
!PARSE-TREE: | | | OutputItem -> Expr = '"pass"'
!PARSE-TREE: | | | | LiteralConstant -> CharLiteralConstant
!PARSE-TREE: | | | | | string = 'pass'
!PARSE-TREE: | EndSubroutineStmt ->

