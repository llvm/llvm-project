!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=45 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=45 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine f00(x)
  integer :: x, i
  !$omp do linear(x)
  do i = 1, 10
    x = x + 1
  enddo
  !$omp end do
end

!UNPARSE: SUBROUTINE f00 (x)
!UNPARSE:  INTEGER x, i
!UNPARSE: !$OMP DO LINEAR(x)
!UNPARSE:  DO i=1_4,10_4
!UNPARSE:   x=x+1_4
!UNPARSE:  END DO
!UNPARSE: !$OMP END DO
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginDirective
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = do
!PARSE-TREE: | OmpClauseList -> OmpClause -> Linear -> OmpLinearClause
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | bool = 'true'
!PARSE-TREE: | Flags = {}

subroutine f01(x)
  integer :: x, i
  !$omp do linear(x : 2)
  do i = 1, 10
    x = x + 2
  enddo
  !$omp end do
end

!UNPARSE: SUBROUTINE f01 (x)
!UNPARSE:  INTEGER x, i
!UNPARSE: !$OMP DO LINEAR(x: 2_4)
!UNPARSE:  DO i=1_4,10_4
!UNPARSE:   x=x+2_4
!UNPARSE:  END DO
!UNPARSE: !$OMP END DO
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginDirective
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = do
!PARSE-TREE: | OmpClauseList -> OmpClause -> Linear -> OmpLinearClause
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | Modifier -> OmpLinearStep -> Scalar -> Integer -> Expr = '2_4'
!PARSE-TREE: | | | LiteralConstant -> IntLiteralConstant = '2'
!PARSE-TREE: | | bool = 'true'
!PARSE-TREE: | Flags = {}
!PARSE-TREE: DoConstruct

subroutine f02(x)
  integer, allocatable :: x
  !$omp declare simd linear(ref(x) : 2)
end

!UNPARSE: SUBROUTINE f02 (x)
!UNPARSE:  INTEGER, ALLOCATABLE :: x
!UNPARSE: !$OMP DECLARE SIMD LINEAR(REF(x): 2_4)
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OpenMPDeclarativeConstruct -> OmpDeclareSimdDirective -> OmpDirectiveSpecification
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = declare simd
!PARSE-TREE: | OmpClauseList -> OmpClause -> Linear -> OmpLinearClause
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | Modifier -> OmpLinearModifier -> Value = Ref
!PARSE-TREE: | | Modifier -> OmpLinearStep -> Scalar -> Integer -> Expr = '2_4'
!PARSE-TREE: | | | LiteralConstant -> IntLiteralConstant = '2'
!PARSE-TREE: | | bool = 'false'
!PARSE-TREE: | Flags = {}

subroutine f03(x, step)
  integer :: step(10)
  integer :: x, i
  !$omp do linear(x : step(3))
  do i = 1, 10
    x = x + 1
  end do
end

!UNPARSE: SUBROUTINE f03 (x, step)
!UNPARSE:  INTEGER step(10_4)
!UNPARSE:  INTEGER x, i
!UNPARSE: !$OMP DO LINEAR(x: step(3_8))
!UNPARSE:  DO i=1_4,10_4
!UNPARSE:    x=x+1_4
!UNPARSE:  END DO
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginDirective
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = do
!PARSE-TREE: | OmpClauseList -> OmpClause -> Linear -> OmpLinearClause
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | Modifier -> OmpLinearStep -> Scalar -> Integer -> Expr = 'step(3_8)'
!PARSE-TREE: | | | Designator -> DataRef -> ArrayElement
!PARSE-TREE: | | | | DataRef -> Name = 'step'
!PARSE-TREE: | | | | SectionSubscript -> Integer -> Expr = '3_4'
!PARSE-TREE: | | | | | LiteralConstant -> IntLiteralConstant = '3'
!PARSE-TREE: | | bool = 'true'
!PARSE-TREE: | Flags = {}
