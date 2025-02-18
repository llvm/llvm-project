!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=52 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=52 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine f00(x)
  integer :: x
  !$omp do linear(x)
  do x = 1, 10
  enddo
  !$omp end do
end

!UNPARSE: SUBROUTINE f00 (x)
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP DO  LINEAR(x)
!UNPARSE:  DO x=1_4,10_4
!UNPARSE:  END DO
!UNPARSE: !$OMP END DO
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginLoopDirective
!PARSE-TREE: | OmpLoopDirective -> llvm::omp::Directive = do
!PARSE-TREE: | OmpClauseList -> OmpClause -> Linear -> OmpLinearClause
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | bool = 'true'
!PARSE-TREE: DoConstruct

subroutine f01(x)
  integer :: x
  !$omp do linear(x : 2)
  do x = 1, 10
  enddo
  !$omp end do
end

!UNPARSE: SUBROUTINE f01 (x)
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP DO  LINEAR(x: 2_4)
!UNPARSE:  DO x=1_4,10_4
!UNPARSE:  END DO
!UNPARSE: !$OMP END DO
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginLoopDirective
!PARSE-TREE: | OmpLoopDirective -> llvm::omp::Directive = do
!PARSE-TREE: | OmpClauseList -> OmpClause -> Linear -> OmpLinearClause
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | Modifier -> OmpStepSimpleModifier -> Scalar -> Integer -> Expr = '2_4'
!PARSE-TREE: | | | LiteralConstant -> IntLiteralConstant = '2'
!PARSE-TREE: | | bool = 'true'
!PARSE-TREE: DoConstruct

subroutine f02(x)
  integer :: x
  !$omp do linear(x : step(3))
  do x = 1, 10
  enddo
  !$omp end do
end

!UNPARSE: SUBROUTINE f02 (x)
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP DO  LINEAR(x: STEP(3_4))
!UNPARSE:  DO x=1_4,10_4
!UNPARSE:  END DO
!UNPARSE: !$OMP END DO
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginLoopDirective
!PARSE-TREE: | OmpLoopDirective -> llvm::omp::Directive = do
!PARSE-TREE: | OmpClauseList -> OmpClause -> Linear -> OmpLinearClause
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | Modifier -> OmpStepComplexModifier -> Scalar -> Integer -> Expr = '3_4'
!PARSE-TREE: | | | LiteralConstant -> IntLiteralConstant = '3'
!PARSE-TREE: | | bool = 'true'
!PARSE-TREE: DoConstruct

subroutine f03(x)
  integer :: x
  !$omp declare simd linear(x : uval)
end

!UNPARSE: SUBROUTINE f03 (x)
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP DECLARE SIMD  LINEAR(x: UVAL)
!UNPARSE: END SUBROUTINE

!PARSE-TREE: SpecificationPart
![...]
!PARSE-TREE: | DeclarationConstruct -> SpecificationConstruct -> OpenMPDeclarativeConstruct -> OpenMPDeclareSimdConstruct
!PARSE-TREE: | | Verbatim
!PARSE-TREE: | | OmpClauseList -> OmpClause -> Linear -> OmpLinearClause
!PARSE-TREE: | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | Modifier -> OmpLinearModifier -> Value = Uval
!PARSE-TREE: | | | bool = 'true'
!PARSE-TREE: ExecutionPart -> Block

subroutine f04(x)
  integer :: x
  !$omp declare simd linear(x : uval, step(3))
end

!UNPARSE: SUBROUTINE f04 (x)
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP DECLARE SIMD  LINEAR(x: UVAL, STEP(3_4))
!UNPARSE: END SUBROUTINE

!PARSE-TREE: SpecificationPart
![...]
!PARSE-TREE: | DeclarationConstruct -> SpecificationConstruct -> OpenMPDeclarativeConstruct -> OpenMPDeclareSimdConstruct
!PARSE-TREE: | | Verbatim
!PARSE-TREE: | | OmpClauseList -> OmpClause -> Linear -> OmpLinearClause
!PARSE-TREE: | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | Modifier -> OmpLinearModifier -> Value = Uval
!PARSE-TREE: | | | Modifier -> OmpStepComplexModifier -> Scalar -> Integer -> Expr = '3_4'
!PARSE-TREE: | | | | LiteralConstant -> IntLiteralConstant = '3'
!PARSE-TREE: | | | bool = 'true'
!PARSE-TREE: ExecutionPart -> Block
