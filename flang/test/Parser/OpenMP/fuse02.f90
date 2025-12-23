! Test the Parse Tree to ensure the OpenMP Loop Transformation Construct Fuse can be constructed on another Fuse

! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=51 %s | FileCheck %s --check-prefix=CHECK-PARSE
! RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=51 %s | FileCheck %s --check-prefix=CHECK-UNPARSE

subroutine fuse_on_fuse
  implicit none
  integer :: I = 10
  integer :: j

  !$omp fuse
    !$omp fuse
      do i = 1, I
        continue
      end do
      do j = 1, I
        continue
      end do
    !$omp end fuse
    do j = 1, I
      continue
    end do
  !$omp end fuse
end subroutine

!CHECK-PARSE: | ExecutionPart -> Block
!CHECK-PARSE-NEXT: | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
!CHECK-PARSE-NEXT: | | | OmpBeginLoopDirective
!CHECK-PARSE-NEXT: | | | | OmpDirectiveName -> llvm::omp::Directive = fuse
!CHECK-PARSE-NEXT: | | | | OmpClauseList ->
!CHECK-PARSE-NEXT: | | | | Flags = None
!CHECK-PARSE-NEXT: | | | Block
!CHECK-PARSE-NEXT: | | | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
!CHECK-PARSE-NEXT: | | | | | OmpBeginLoopDirective
!CHECK-PARSE-NEXT: | | | | | | OmpDirectiveName -> llvm::omp::Directive = fuse
!CHECK-PARSE-NEXT: | | | | | | OmpClauseList ->
!CHECK-PARSE-NEXT: | | | | | | Flags = None
!CHECK-PARSE-NEXT: | | | | | Block
!CHECK-PARSE-NEXT: | | | | | | ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct
!CHECK-PARSE-NEXT: | | | | | | | NonLabelDoStmt
!CHECK-PARSE-NEXT: | | | | | | | | LoopControl -> LoopBounds
!CHECK-PARSE-NEXT: | | | | | | | | | Scalar -> Name = 'i'
!CHECK-PARSE-NEXT: | | | | | | | | | Scalar -> Expr = '1_4'
!CHECK-PARSE-NEXT: | | | | | | | | | | LiteralConstant -> IntLiteralConstant = '1'
!CHECK-PARSE-NEXT: | | | | | | | | | Scalar -> Expr = 'i'
!CHECK-PARSE-NEXT: | | | | | | | | | | Designator -> DataRef -> Name = 'i'
!CHECK-PARSE-NEXT: | | | | | | | Block
!CHECK-PARSE-NEXT: | | | | | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> ContinueStmt
!CHECK-PARSE-NEXT: | | | | | | | EndDoStmt ->
!CHECK-PARSE-NEXT: | | | | | | ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct
!CHECK-PARSE-NEXT: | | | | | | | NonLabelDoStmt
!CHECK-PARSE-NEXT: | | | | | | | | LoopControl -> LoopBounds
!CHECK-PARSE-NEXT: | | | | | | | | | Scalar -> Name = 'j'
!CHECK-PARSE-NEXT: | | | | | | | | | Scalar -> Expr = '1_4'
!CHECK-PARSE-NEXT: | | | | | | | | | | LiteralConstant -> IntLiteralConstant = '1'
!CHECK-PARSE-NEXT: | | | | | | | | | Scalar -> Expr = 'i'
!CHECK-PARSE-NEXT: | | | | | | | | | | Designator -> DataRef -> Name = 'i'
!CHECK-PARSE-NEXT: | | | | | | | Block
!CHECK-PARSE-NEXT: | | | | | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> ContinueStmt
!CHECK-PARSE-NEXT: | | | | | | | EndDoStmt ->
!CHECK-PARSE-NEXT: | | | | | OmpEndLoopDirective
!CHECK-PARSE-NEXT: | | | | | | OmpDirectiveName -> llvm::omp::Directive = fuse
!CHECK-PARSE-NEXT: | | | | | | OmpClauseList ->
!CHECK-PARSE-NEXT: | | | | | | Flags = None
!CHECK-PARSE-NEXT: | | | | ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct
!CHECK-PARSE-NEXT: | | | | | NonLabelDoStmt
!CHECK-PARSE-NEXT: | | | | | | LoopControl -> LoopBounds
!CHECK-PARSE-NEXT: | | | | | | | Scalar -> Name = 'j'
!CHECK-PARSE-NEXT: | | | | | | | Scalar -> Expr = '1_4'
!CHECK-PARSE-NEXT: | | | | | | | | LiteralConstant -> IntLiteralConstant = '1'
!CHECK-PARSE-NEXT: | | | | | | | Scalar -> Expr = 'i'
!CHECK-PARSE-NEXT: | | | | | | | | Designator -> DataRef -> Name = 'i'
!CHECK-PARSE-NEXT: | | | | | Block
!CHECK-PARSE-NEXT: | | | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> ContinueStmt
!CHECK-PARSE-NEXT: | | | | | EndDoStmt ->
!CHECK-PARSE-NEXT: | | | OmpEndLoopDirective
!CHECK-PARSE-NEXT: | | | | OmpDirectiveName -> llvm::omp::Directive = fuse
!CHECK-PARSE-NEXT: | | | | OmpClauseList ->
!CHECK-PARSE-NEXT: | | | | Flags = None

!CHECK-UNPARSE: SUBROUTINE fuse_on_fuse
!CHECK-UNPARSE-NEXT:  IMPLICIT NONE
!CHECK-UNPARSE-NEXT:  INTEGER :: i = 10_4
!CHECK-UNPARSE-NEXT:  INTEGER j
!CHECK-UNPARSE-NEXT: !$OMP FUSE
!CHECK-UNPARSE-NEXT: !$OMP FUSE
!CHECK-UNPARSE-NEXT:  DO i=1_4,i
!CHECK-UNPARSE-NEXT:    CONTINUE
!CHECK-UNPARSE-NEXT:  END DO
!CHECK-UNPARSE-NEXT:  DO j=1_4,i
!CHECK-UNPARSE-NEXT:    CONTINUE
!CHECK-UNPARSE-NEXT:  END DO
!CHECK-UNPARSE-NEXT: !$OMP END FUSE
!CHECK-UNPARSE-NEXT:  DO j=1_4,i
!CHECK-UNPARSE-NEXT:    CONTINUE
!CHECK-UNPARSE-NEXT:  END DO
!CHECK-UNPARSE-NEXT: !$OMP END FUSE
