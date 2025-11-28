! This reproducer hit an issue where when finding directive's, and end directive's would iterate over the block.end()
! so Flang would crash. We should be able to parse this subroutine without flang crashing.
! Reported in https://github.com/llvm/llvm-project/issues/147309 and https://github.com/llvm/llvm-project/pull/145917#issuecomment-3041570824

!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=51 %s | FileCheck %s --check-prefix=CHECK-PARSE
!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=51 %s | FileCheck %s --check-prefix=CHECK-UNPARSE

subroutine loop_transformation_construct7
implicit none
real(kind=8), dimension(1:10, 2) :: a
integer :: b,c

!$omp target teams distribute parallel do collapse(2) private(b)
do b = 1, 10
  do c = 1, 10
    a(b, 2) = a(c, 1)
  end do
end do
end subroutine

!CHECK-PARSE: | ExecutionPart -> Block
!CHECK-PARSE-NEXT: | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
!CHECK-PARSE-NEXT: | | | OmpBeginLoopDirective
!CHECK-PARSE-NEXT: | | | | OmpDirectiveName -> llvm::omp::Directive = target teams distribute parallel do
!CHECK-PARSE-NEXT: | | | | OmpClauseList -> OmpClause -> Collapse -> Scalar -> Integer -> Constant -> Expr = '2_4'
!CHECK-PARSE-NEXT: | | | | | LiteralConstant -> IntLiteralConstant = '2'
!CHECK-PARSE-NEXT: | | | | OmpClause -> Private -> OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'b'
!CHECK-PARSE-NEXT: | | | | Flags = None
!CHECK-PARSE-NEXT: | | | Block
!CHECK-PARSE-NEXT: | | | | ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct
!CHECK-PARSE-NEXT: | | | | | NonLabelDoStmt
!CHECK-PARSE-NEXT: | | | | | | LoopControl -> LoopBounds
!CHECK-PARSE-NEXT: | | | | | | | Scalar -> Name = 'b'
!CHECK-PARSE-NEXT: | | | | | | | Scalar -> Expr = '1_4'
!CHECK-PARSE-NEXT: | | | | | | | | LiteralConstant -> IntLiteralConstant = '1'
!CHECK-PARSE-NEXT: | | | | | | | Scalar -> Expr = '10_4'
!CHECK-PARSE-NEXT: | | | | | | | | LiteralConstant -> IntLiteralConstant = '10'
!CHECK-PARSE-NEXT: | | | | | Block
!CHECK-PARSE-NEXT: | | | | | | ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct
!CHECK-PARSE-NEXT: | | | | | | | NonLabelDoStmt
!CHECK-PARSE-NEXT: | | | | | | | | LoopControl -> LoopBounds
!CHECK-PARSE-NEXT: | | | | | | | | | Scalar -> Name = 'c'
!CHECK-PARSE-NEXT: | | | | | | | | | Scalar -> Expr = '1_4'
!CHECK-PARSE-NEXT: | | | | | | | | | | LiteralConstant -> IntLiteralConstant = '1'
!CHECK-PARSE-NEXT: | | | | | | | | | Scalar -> Expr = '10_4'
!CHECK-PARSE-NEXT: | | | | | | | | | | LiteralConstant -> IntLiteralConstant = '10'
!CHECK-PARSE-NEXT: | | | | | | | Block
!CHECK-PARSE-NEXT: | | | | | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'a(int(b,kind=8),2_8)=a(int(c,kind=8),1_8)'
!CHECK-PARSE-NEXT: | | | | | | | | | Variable = 'a(int(b,kind=8),2_8)'
!CHECK-PARSE-NEXT: | | | | | | | | | | Designator -> DataRef -> ArrayElement
!CHECK-PARSE-NEXT: | | | | | | | | | | | DataRef -> Name = 'a'
!CHECK-PARSE-NEXT: | | | | | | | | | | | SectionSubscript -> Integer -> Expr = 'b'
!CHECK-PARSE-NEXT: | | | | | | | | | | | | Designator -> DataRef -> Name = 'b'
!CHECK-PARSE-NEXT: | | | | | | | | | | | SectionSubscript -> Integer -> Expr = '2_4'
!CHECK-PARSE-NEXT: | | | | | | | | | | | | LiteralConstant -> IntLiteralConstant = '2'
!CHECK-PARSE-NEXT: | | | | | | | | | Expr = 'a(int(c,kind=8),1_8)'
!CHECK-PARSE-NEXT: | | | | | | | | | | Designator -> DataRef -> ArrayElement
!CHECK-PARSE-NEXT: | | | | | | | | | | | DataRef -> Name = 'a'
!CHECK-PARSE-NEXT: | | | | | | | | | | | SectionSubscript -> Integer -> Expr = 'c'
!CHECK-PARSE-NEXT: | | | | | | | | | | | | Designator -> DataRef -> Name = 'c'
!CHECK-PARSE-NEXT: | | | | | | | | | | | SectionSubscript -> Integer -> Expr = '1_4'
!CHECK-PARSE-NEXT: | | | | | | | | | | | | LiteralConstant -> IntLiteralConstant = '1'
!CHECK-PARSE-NEXT: | | | | | | | EndDoStmt ->
!CHECK-PARSE-NEXT: | | | | | EndDoStmt ->
!CHECK-PARSE-NEXT: | EndSubroutineStmt ->

!CHECK-UNPARSE: SUBROUTINE loop_transformation_construct7
!CHECK-UNPARSE-NEXT:  IMPLICIT NONE
!CHECK-UNPARSE-NEXT:  REAL(KIND=8_4), DIMENSION(1_4:10_4,2_4) :: a
!CHECK-UNPARSE-NEXT:  INTEGER b, c
!CHECK-UNPARSE-NEXT: !$OMP TARGET TEAMS DISTRIBUTE PARALLEL DO  COLLAPSE(2_4) PRIVATE(b)
!CHECK-UNPARSE-NEXT:  DO b=1_4,10_4
!CHECK-UNPARSE-NEXT:   DO c=1_4,10_4
!CHECK-UNPARSE-NEXT:     a(int(b,kind=8),2_8)=a(int(c,kind=8),1_8)
!CHECK-UNPARSE-NEXT:   END DO
!CHECK-UNPARSE-NEXT:  END DO
!CHECK-UNPARSE-NEXT: END SUBROUTINE
