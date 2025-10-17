! Test the Parse Tree to ensure the OpenMP Loop Transformation Constructs nest correctly with multiple nested loops.

! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=51 %s | FileCheck %s --check-prefix=CHECK-PARSE
! RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=51 %s | FileCheck %s --check-prefix=CHECK-UNPARSE

subroutine loop_transformation_construct
  implicit none
  integer :: I = 10
  integer :: x
  integer :: y(I)

  !$omp do
  !$omp unroll
  !$omp tile sizes(2)
  do i = 1, I
    y(i) = y(i) * 5
  end do
  !$omp end tile
  !$omp end unroll
  !$omp end do
end subroutine

!CHECK-PARSE: | ExecutionPart -> Block
!CHECK-PARSE-NEXT: | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
!CHECK-PARSE-NEXT: | | | OmpBeginLoopDirective
!CHECK-PARSE-NEXT: | | | | OmpDirectiveName -> llvm::omp::Directive = do
!CHECK-PARSE-NEXT: | | | | OmpClauseList ->
!CHECK-PARSE-NEXT: | | | | Flags = None
!CHECK-PARSE-NEXT: | | | OpenMPLoopConstruct
!CHECK-PARSE-NEXT: | | | | OmpBeginLoopDirective
!CHECK-PARSE-NEXT: | | | | | OmpDirectiveName -> llvm::omp::Directive = unroll
!CHECK-PARSE-NEXT: | | | | | OmpClauseList ->
!CHECK-PARSE-NEXT: | | | | | Flags = None
!CHECK-PARSE-NEXT: | | | | OpenMPLoopConstruct
!CHECK-PARSE-NEXT: | | | | | OmpBeginLoopDirective
!CHECK-PARSE-NEXT: | | | | | | OmpDirectiveName -> llvm::omp::Directive = tile
!CHECK-PARSE-NEXT: | | | | | | OmpClauseList -> OmpClause -> Sizes -> Scalar -> Integer -> Expr = '2_4'
!CHECK-PARSE-NEXT: | | | | | | | LiteralConstant -> IntLiteralConstant = '2'
!CHECK-PARSE-NEXT: | | | | | | Flags = None
!CHECK-PARSE-NEXT: | | | | | DoConstruct
!CHECK-PARSE-NEXT: | | | | | | NonLabelDoStmt
!CHECK-PARSE-NEXT: | | | | | | | LoopControl -> LoopBounds
!CHECK-PARSE-NEXT: | | | | | | | | Scalar -> Name = 'i'
!CHECK-PARSE-NEXT: | | | | | | | | Scalar -> Expr = '1_4'
!CHECK-PARSE-NEXT: | | | | | | | | | LiteralConstant -> IntLiteralConstant = '1'
!CHECK-PARSE-NEXT: | | | | | | | | Scalar -> Expr = 'i'
!CHECK-PARSE-NEXT: | | | | | | | | | Designator -> DataRef -> Name = 'i'
!CHECK-PARSE-NEXT: | | | | | | Block
!CHECK-PARSE-NEXT: | | | | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'y(int(i,kind=8))=5_4*y(int(i,kind=8))'
!CHECK-PARSE-NEXT: | | | | | | | | Variable = 'y(int(i,kind=8))'
!CHECK-PARSE-NEXT: | | | | | | | | | Designator -> DataRef -> ArrayElement
!CHECK-PARSE-NEXT: | | | | | | | | | | DataRef -> Name = 'y'
!CHECK-PARSE-NEXT: | | | | | | | | | | SectionSubscript -> Integer -> Expr = 'i'
!CHECK-PARSE-NEXT: | | | | | | | | | | | Designator -> DataRef -> Name = 'i'
!CHECK-PARSE-NEXT: | | | | | | | | Expr = '5_4*y(int(i,kind=8))'
!CHECK-PARSE-NEXT: | | | | | | | | | Multiply
!CHECK-PARSE-NEXT: | | | | | | | | | | Expr = 'y(int(i,kind=8))'
!CHECK-PARSE-NEXT: | | | | | | | | | | | Designator -> DataRef -> ArrayElement
!CHECK-PARSE-NEXT: | | | | | | | | | | | | DataRef -> Name = 'y'
!CHECK-PARSE-NEXT: | | | | | | | | | | | | SectionSubscript -> Integer -> Expr = 'i'
!CHECK-PARSE-NEXT: | | | | | | | | | | | | | Designator -> DataRef -> Name = 'i'
!CHECK-PARSE-NEXT: | | | | | | | | | | Expr = '5_4'
!CHECK-PARSE-NEXT: | | | | | | | | | | | LiteralConstant -> IntLiteralConstant = '5'
!CHECK-PARSE-NEXT: | | | | | | EndDoStmt ->
!CHECK-PARSE-NEXT: | | | | | OmpEndLoopDirective
!CHECK-PARSE-NEXT: | | | | | | OmpDirectiveName -> llvm::omp::Directive = tile
!CHECK-PARSE-NEXT: | | | | | | OmpClauseList ->
!CHECK-PARSE-NEXT: | | | | | | Flags = None
!CHECK-PARSE-NEXT: | | | | OmpEndLoopDirective
!CHECK-PARSE-NEXT: | | | | | OmpDirectiveName -> llvm::omp::Directive = unroll
!CHECK-PARSE-NEXT: | | | | | OmpClauseList ->
!CHECK-PARSE-NEXT: | | | | | Flags = None
!CHECK-PARSE-NEXT: | | | OmpEndLoopDirective
!CHECK-PARSE-NEXT: | | | | OmpDirectiveName -> llvm::omp::Directive = do
!CHECK-PARSE-NEXT: | | | | OmpClauseList ->
!CHECK-PARSE-NEXT: | | | | Flags = None

!CHECK-UNPARSE: SUBROUTINE loop_transformation_construct
!CHECK-UNPARSE-NEXT:  IMPLICIT NONE
!CHECK-UNPARSE-NEXT:  INTEGER :: i = 10_4
!CHECK-UNPARSE-NEXT:  INTEGER x
!CHECK-UNPARSE-NEXT:  INTEGER y(i)
!CHECK-UNPARSE-NEXT: !$OMP DO
!CHECK-UNPARSE-NEXT: !$OMP UNROLL
!CHECK-UNPARSE-NEXT: !$OMP TILE
!CHECK-UNPARSE-NEXT:  DO i=1_4,i
!CHECK-UNPARSE-NEXT:    y(int(i,kind=8))=5_4*y(int(i,kind=8))
!CHECK-UNPARSE-NEXT:  END DO
!CHECK-UNPARSE-NEXT: !$OMP END TILE
!CHECK-UNPARSE-NEXT: !$OMP END UNROLL
!CHECK-UNPARSE-NEXT: !$OMP END DO
!CHECK-UNPARSE-NEXT: END SUBROUTINE
