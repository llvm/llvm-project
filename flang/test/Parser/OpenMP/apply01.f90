!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=60 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=60 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine apply_1(x)
  implicit none
  integer :: x, i, j

  !$omp interchange apply(nothing, reverse)
  do i = 1,10
    do j = 1,10
      x = x + 1
    end do
  end do
end

!UNPARSE: !$OMP INTERCHANGE APPLY(NOTHING, REVERSE)
!UNPARSE:  DO i=1_4,10_4
!UNPARSE:   DO j=1_4,10_4
!UNPARSE:    x=x+1_4
!UNPARSE:   END DO
!UNPARSE:  END DO

!PARSE-TREE: OmpBeginDirective
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = interchange
!PARSE-TREE: | OmpClauseList -> OmpClause -> Apply -> OmpApplyClause
!PARSE-TREE: | | OmpDirectiveSpecification
!PARSE-TREE: | | | OmpDirectiveName -> llvm::omp::Directive = nothing
!PARSE-TREE: | | | OmpClauseList ->
!PARSE-TREE: | | | Flags = {}
!PARSE-TREE: | | OmpDirectiveSpecification
!PARSE-TREE: | | | OmpDirectiveName -> llvm::omp::Directive = reverse
!PARSE-TREE: | | | OmpClauseList ->
!PARSE-TREE: | | | Flags = {}

subroutine apply_modifier(x)
  implicit none
  integer :: x, i

  !$omp tile sizes(2) apply(grid: reverse)
  do i = 1, 10
    x = x + 1
  end do
end

!UNPARSE: !$OMP TILE SIZES(2_4) APPLY(GRID: REVERSE)
!UNPARSE:  DO i=1_4,10_4
!UNPARSE:    x=x+1_4
!UNPARSE:  END DO

!PARSE-TREE: OmpBeginDirective
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = tile
!PARSE-TREE: | OmpClauseList -> OmpClause -> Sizes -> Scalar -> Integer -> Expr = '2_4'
!PARSE-TREE: | | LiteralConstant -> IntLiteralConstant = '2'
!PARSE-TREE: | OmpClause -> Apply -> OmpApplyClause
!PARSE-TREE: | | Modifier -> OmpLoopModifier
!PARSE-TREE: | | | llvm::omp::LoopModifier = grid
!PARSE-TREE: | | OmpDirectiveSpecification
!PARSE-TREE: | | | OmpDirectiveName -> llvm::omp::Directive = reverse
!PARSE-TREE: | | | OmpClauseList ->
!PARSE-TREE: | | | Flags = {}

subroutine apply_2_clauses(x)
  implicit none
  integer :: x, i

  !$omp tile sizes(2) apply(intratile: unroll) apply(grid: reverse)
  do i = 1, 10
    x = x + 1
  end do
end

!UNPARSE: !$OMP TILE SIZES(2_4) APPLY(INTRATILE: UNROLL) APPLY(GRID: REVERSE)
!UNPARSE:  DO i=1_4,10_4
!UNPARSE:    x=x+1_4
!UNPARSE:  END DO

!PARSE-TREE: OmpBeginDirective
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = tile
!PARSE-TREE: | OmpClauseList -> OmpClause -> Sizes -> Scalar -> Integer -> Expr = '2_4'
!PARSE-TREE: | | LiteralConstant -> IntLiteralConstant = '2'
!PARSE-TREE: | OmpClause -> Apply -> OmpApplyClause
!PARSE-TREE: | | Modifier -> OmpLoopModifier
!PARSE-TREE: | | | llvm::omp::LoopModifier = intratile
!PARSE-TREE: | | OmpDirectiveSpecification
!PARSE-TREE: | | | OmpDirectiveName -> llvm::omp::Directive = unroll
!PARSE-TREE: | | | OmpClauseList ->
!PARSE-TREE: | | | Flags = {}
!PARSE-TREE: | OmpClause -> Apply -> OmpApplyClause
!PARSE-TREE: | | Modifier -> OmpLoopModifier
!PARSE-TREE: | | | llvm::omp::LoopModifier = grid
!PARSE-TREE: | | OmpDirectiveSpecification
!PARSE-TREE: | | | OmpDirectiveName -> llvm::omp::Directive = reverse
!PARSE-TREE: | | | OmpClauseList ->
!PARSE-TREE: | | | Flags = {}


subroutine apply_inside_apply(x)
  implicit none
  integer :: x, i, j

  !$omp fuse apply(fused: tile sizes(2) apply(grid: reverse))
  do i = 1, 10
    x = x + 1
  end do
  do j = 1, 10
    x = x - 1
  end do
  !$omp end fuse
end

!UNPARSE: !$OMP FUSE APPLY(FUSED: TILE SIZES(2_4) APPLY(GRID: REVERSE))
!UNPARSE:  DO i=1_4,10_4
!UNPARSE:    x=x+1_4
!UNPARSE:  END DO
!UNPARSE:  DO j=1_4,10_4
!UNPARSE:    x=x-1_4
!UNPARSE:  END DO
!UNPARSE: !$OMP END FUSE

!PARSE-TREE: OmpBeginDirective
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = fuse
!PARSE-TREE: | OmpClauseList -> OmpClause -> Apply -> OmpApplyClause
!PARSE-TREE: | | Modifier -> OmpLoopModifier
!PARSE-TREE: | | | llvm::omp::LoopModifier = fused
!PARSE-TREE: | | OmpDirectiveSpecification
!PARSE-TREE: | | | OmpDirectiveName -> llvm::omp::Directive = tile
!PARSE-TREE: | | | OmpClauseList -> OmpClause -> Sizes -> Scalar -> Integer -> Expr = '2_4'
!PARSE-TREE: | | | | LiteralConstant -> IntLiteralConstant = '2'
!PARSE-TREE: | | | OmpClause -> Apply -> OmpApplyClause
!PARSE-TREE: | | | | Modifier -> OmpLoopModifier
!PARSE-TREE: | | | | | llvm::omp::LoopModifier = grid
!PARSE-TREE: | | | | OmpDirectiveSpecification
!PARSE-TREE: | | | | | OmpDirectiveName -> llvm::omp::Directive = reverse
!PARSE-TREE: | | | | | OmpClauseList ->
!PARSE-TREE: | | | | | Flags = {}
!PARSE-TREE: | | | Flags = {}

