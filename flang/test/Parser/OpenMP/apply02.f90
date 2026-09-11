!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=60 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=60 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine apply_all(x)
  implicit none
  integer :: x, i, j

  !$omp interchange apply(interchanged(1,2): nothing, reverse)
  do i = 1,10
    do j = 1,10
      x = x + 1
    end do
  end do
end

!UNPARSE: !$OMP INTERCHANGE APPLY(INTERCHANGED(1_4, 2_4): NOTHING, REVERSE)
!UNPARSE:  DO i=1_4,10_4
!UNPARSE:   DO j=1_4,10_4
!UNPARSE:     x=x+1_4
!UNPARSE:   END DO
!UNPARSE:  END DO

!PARSE-TREE: OmpBeginDirective
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = interchange
!PARSE-TREE: | OmpClauseList -> OmpClause -> Apply -> OmpApplyClause
!PARSE-TREE: | | Modifier -> OmpLoopModifier
!PARSE-TREE: | | | llvm::omp::LoopModifier = interchanged
!PARSE-TREE: | | | Scalar -> Integer -> Constant -> Expr = '1_4'
!PARSE-TREE: | | | | LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | | | Scalar -> Integer -> Constant -> Expr = '2_4'
!PARSE-TREE: | | | | LiteralConstant -> IntLiteralConstant = '2'
!PARSE-TREE: | | OmpDirectiveSpecification
!PARSE-TREE: | | | OmpDirectiveName -> llvm::omp::Directive = nothing
!PARSE-TREE: | | | OmpClauseList ->
!PARSE-TREE: | | | Flags = {}
!PARSE-TREE: | | OmpDirectiveSpecification
!PARSE-TREE: | | | OmpDirectiveName -> llvm::omp::Directive = reverse
!PARSE-TREE: | | | OmpClauseList ->
!PARSE-TREE: | | | Flags = {}

subroutine apply_one(x)
  implicit none
  integer :: x, i, j

  !$omp interchange apply(interchanged(2): reverse)
  do i = 1,10
    do j = 1,10
      x = x + 1
    end do
  end do
end

!UNPARSE: !$OMP INTERCHANGE APPLY(INTERCHANGED(2_4): REVERSE)
!UNPARSE:  DO i=1_4,10_4
!UNPARSE:   DO j=1_4,10_4
!UNPARSE:     x=x+1_4
!UNPARSE:   END DO
!UNPARSE:  END DO

!PARSE-TREE: OmpBeginDirective
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = interchange
!PARSE-TREE: | OmpClauseList -> OmpClause -> Apply -> OmpApplyClause
!PARSE-TREE: | | Modifier -> OmpLoopModifier
!PARSE-TREE: | | | llvm::omp::LoopModifier = interchanged
!PARSE-TREE: | | | Scalar -> Integer -> Constant -> Expr = '2_4'
!PARSE-TREE: | | | | LiteralConstant -> IntLiteralConstant = '2'
!PARSE-TREE: | | OmpDirectiveSpecification
!PARSE-TREE: | | | OmpDirectiveName -> llvm::omp::Directive = reverse
!PARSE-TREE: | | | OmpClauseList ->
!PARSE-TREE: | | | Flags = {}

subroutine apply_inside_apply(x)
  implicit none
  integer :: x, i, j

  !$omp tile sizes(2,2) apply(grid(1): interchange apply(interchanged(2): reverse))
  do i = 1,10
    do j = 1,10
      x = x + 1
    end do
  end do
end

!UNPARSE: !$OMP TILE SIZES(2_4,2_4) APPLY(GRID(1_4): INTERCHANGE APPLY(INTERCHANGED(2_4): REVERSE))
!UNPARSE:  DO i=1_4,10_4
!UNPARSE:   DO j=1_4,10_4
!UNPARSE:     x=x+1_4
!UNPARSE:   END DO
!UNPARSE:  END DO

!PARSE-TREE: OmpBeginDirective
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = tile
!PARSE-TREE: | OmpClauseList -> OmpClause -> Sizes -> Scalar -> Integer -> Expr = '2_4'
!PARSE-TREE: | | LiteralConstant -> IntLiteralConstant = '2'
!PARSE-TREE: | Scalar -> Integer -> Expr = '2_4'
!PARSE-TREE: | | LiteralConstant -> IntLiteralConstant = '2'
!PARSE-TREE: | OmpClause -> Apply -> OmpApplyClause
!PARSE-TREE: | | Modifier -> OmpLoopModifier
!PARSE-TREE: | | | llvm::omp::LoopModifier = grid
!PARSE-TREE: | | | Scalar -> Integer -> Constant -> Expr = '1_4'
!PARSE-TREE: | | | | LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | | OmpDirectiveSpecification
!PARSE-TREE: | | | OmpDirectiveName -> llvm::omp::Directive = interchange
!PARSE-TREE: | | | OmpClauseList -> OmpClause -> Apply -> OmpApplyClause
!PARSE-TREE: | | | | Modifier -> OmpLoopModifier
!PARSE-TREE: | | | | | llvm::omp::LoopModifier = interchanged
!PARSE-TREE: | | | | | Scalar -> Integer -> Constant -> Expr = '2_4'
!PARSE-TREE: | | | | | | LiteralConstant -> IntLiteralConstant = '2'
!PARSE-TREE: | | | | OmpDirectiveSpecification
!PARSE-TREE: | | | | | OmpDirectiveName -> llvm::omp::Directive = reverse
!PARSE-TREE: | | | | | OmpClauseList ->
!PARSE-TREE: | | | | | Flags = {}
!PARSE-TREE: | | | Flags = {}

