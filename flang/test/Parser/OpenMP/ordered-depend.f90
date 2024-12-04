!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=45 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=45 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine f00(x)
  integer :: x(10, 10)
  !$omp do ordered(2)
  do i = 1, 10
    do j = 1, 10
      !$omp ordered depend(source)
      x(i, j) = i + j
    enddo
  enddo
  !$omp end do
end

!UNPARSE: SUBROUTINE f00 (x)
!UNPARSE:  INTEGER x(10_4,10_4)
!UNPARSE: !$OMP DO  ORDERED(2_4)
!UNPARSE:  DO i=1_4,10_4
!UNPARSE:   DO j=1_4,10_4
!UNPARSE: !$OMP ORDERED  DEPEND(SOURCE)
!UNPARSE:     x(int(i,kind=8),int(j,kind=8))=i+j
!UNPARSE:   END DO
!UNPARSE:  END DO
!UNPARSE: !$OMP END DO
!UNPARSE: END SUBROUTINE

!PARSE-TREE-LABEL: ProgramUnit -> SubroutineSubprogram
!PARSE-TREE: OmpBeginLoopDirective
!PARSE-TREE: | OmpLoopDirective -> llvm::omp::Directive = do
!PARSE-TREE: | OmpClauseList -> OmpClause -> Ordered -> Scalar -> Integer -> Constant -> Expr = '2_4'
!PARSE-TREE: | | LiteralConstant -> IntLiteralConstant = '2'
![...]
!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPSimpleStandaloneConstruct
!PARSE-TREE: | OmpSimpleStandaloneDirective -> llvm::omp::Directive = ordered
!PARSE-TREE: | OmpClauseList -> OmpClause -> Depend -> OmpDependClause -> OmpDoacross -> Source

subroutine f01(x)
  integer :: x(10, 10)
  !$omp do ordered(2)
  do i = 1, 10
    do j = 1, 10
      !$omp ordered depend(sink: i+1, j-2), depend(sink: i, j+3)
      x(i, j) = i + j
    enddo
  enddo
  !$omp end do
end

!UNPARSE: SUBROUTINE f01 (x)
!UNPARSE:  INTEGER x(10_4,10_4)
!UNPARSE: !$OMP DO  ORDERED(2_4)
!UNPARSE:  DO i=1_4,10_4
!UNPARSE:   DO j=1_4,10_4
!UNPARSE: !$OMP ORDERED  DEPEND(SINK: i+1_4, j-2_4) DEPEND(SINK: i, j+3_4)
!UNPARSE:     x(int(i,kind=8),int(j,kind=8))=i+j
!UNPARSE:   END DO
!UNPARSE:  END DO
!UNPARSE: !$OMP END DO
!UNPARSE: END SUBROUTINE

!PARSE-TREE-LABEL: ProgramUnit -> SubroutineSubprogram
!PARSE-TREE: OmpBeginLoopDirective
!PARSE-TREE: | OmpLoopDirective -> llvm::omp::Directive = do
!PARSE-TREE: | OmpClauseList -> OmpClause -> Ordered -> Scalar -> Integer -> Constant -> Expr = '2_4'
!PARSE-TREE: | | LiteralConstant -> IntLiteralConstant = '2'
![...]
!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPSimpleStandaloneConstruct
!PARSE-TREE: | OmpSimpleStandaloneDirective -> llvm::omp::Directive = ordered
!PARSE-TREE: | OmpClauseList -> OmpClause -> Depend -> OmpDependClause -> OmpDoacross -> Sink -> OmpIterationVector -> OmpIteration
!PARSE-TREE: | | Name = 'i'
!PARSE-TREE: | | OmpIterationOffset
!PARSE-TREE: | | | DefinedOperator -> IntrinsicOperator = Add
!PARSE-TREE: | | | Scalar -> Integer -> Constant -> Expr = '1_4'
!PARSE-TREE: | | | | LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | OmpIteration
!PARSE-TREE: | | Name = 'j'
!PARSE-TREE: | | OmpIterationOffset
!PARSE-TREE: | | | DefinedOperator -> IntrinsicOperator = Subtract
!PARSE-TREE: | | | Scalar -> Integer -> Constant -> Expr = '2_4'
!PARSE-TREE: | | | | LiteralConstant -> IntLiteralConstant = '2'
!PARSE-TREE: | OmpClause -> Depend -> OmpDependClause -> OmpDoacross -> Sink -> OmpIterationVector -> OmpIteration
!PARSE-TREE: | | Name = 'i'
!PARSE-TREE: | OmpIteration
!PARSE-TREE: | | Name = 'j'
!PARSE-TREE: | | OmpIterationOffset
!PARSE-TREE: | | | DefinedOperator -> IntrinsicOperator = Add
!PARSE-TREE: | | | Scalar -> Integer -> Constant -> Expr = '3_4'
!PARSE-TREE: | | | | LiteralConstant -> IntLiteralConstant = '3'

