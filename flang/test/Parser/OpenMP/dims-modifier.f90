!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=61 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=61 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine f00
  !$omp teams num_teams(dims(2): 10, 4)
  !$omp end teams
end

!UNPARSE: SUBROUTINE f00
!UNPARSE: !$OMP TEAMS NUM_TEAMS(DIMS(2_4):10_4, 4_4)
!UNPARSE: !$OMP END TEAMS
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginDirective
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = teams
!PARSE-TREE: | OmpClauseList -> OmpClause -> NumTeams -> OmpNumTeamsClause
!PARSE-TREE: | | Modifier -> OmpDimsModifier -> Scalar -> Integer -> Constant -> Expr = '2_4'
!PARSE-TREE: | | | LiteralConstant -> IntLiteralConstant = '2'
!PARSE-TREE: | | Scalar -> Integer -> Expr = '10_4'
!PARSE-TREE: | | | LiteralConstant -> IntLiteralConstant = '10'
!PARSE-TREE: | | Scalar -> Integer -> Expr = '4_4'
!PARSE-TREE: | | | LiteralConstant -> IntLiteralConstant = '4'
!PARSE-TREE: | Flags = {}


subroutine f01
  !$omp teams num_teams(dims(2), 3: 10, 4)
  !$omp end teams
end

!UNPARSE: SUBROUTINE f01
!UNPARSE: !$OMP TEAMS NUM_TEAMS(DIMS(2_4), 3_4:10_4, 4_4)
!UNPARSE: !$OMP END TEAMS
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginDirective
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = teams
!PARSE-TREE: | OmpClauseList -> OmpClause -> NumTeams -> OmpNumTeamsClause
!PARSE-TREE: | | Modifier -> OmpDimsModifier -> Scalar -> Integer -> Constant -> Expr = '2_4'
!PARSE-TREE: | | | LiteralConstant -> IntLiteralConstant = '2'
!PARSE-TREE: | | Modifier -> OmpLowerBound -> Scalar -> Integer -> Expr = '3_4'
!PARSE-TREE: | | | LiteralConstant -> IntLiteralConstant = '3'
!PARSE-TREE: | | Scalar -> Integer -> Expr = '10_4'
!PARSE-TREE: | | | LiteralConstant -> IntLiteralConstant = '10'
!PARSE-TREE: | | Scalar -> Integer -> Expr = '4_4'
!PARSE-TREE: | | | LiteralConstant -> IntLiteralConstant = '4'
!PARSE-TREE: | Flags = {}


subroutine f02
  !$omp teams thread_limit(dims(3): 16)
  !$omp end teams
end

!UNPARSE: SUBROUTINE f02
!UNPARSE: !$OMP TEAMS THREAD_LIMIT(DIMS(3_4):16_4)
!UNPARSE: !$OMP END TEAMS
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginDirective
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = teams
!PARSE-TREE: | OmpClauseList -> OmpClause -> ThreadLimit -> OmpThreadLimitClause
!PARSE-TREE: | | Modifier -> OmpDimsModifier -> Scalar -> Integer -> Constant -> Expr = '3_4'
!PARSE-TREE: | | | LiteralConstant -> IntLiteralConstant = '3'
!PARSE-TREE: | | Scalar -> Integer -> Expr = '16_4'
!PARSE-TREE: | | | LiteralConstant -> IntLiteralConstant = '16'
!PARSE-TREE: | Flags = {}


subroutine f03
  !$omp parallel num_threads(dims(4): 4, 5, 6, 7)
  !$omp end parallel
end

!UNPARSE: SUBROUTINE f03
!UNPARSE: !$OMP PARALLEL NUM_THREADS(DIMS(4_4):4_4, 5_4, 6_4, 7_4)
!UNPARSE: !$OMP END PARALLEL
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginDirective
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = parallel
!PARSE-TREE: | OmpClauseList -> OmpClause -> NumThreads -> OmpNumThreadsClause
!PARSE-TREE: | | Modifier -> OmpDimsModifier -> Scalar -> Integer -> Constant -> Expr = '4_4'
!PARSE-TREE: | | | LiteralConstant -> IntLiteralConstant = '4'
!PARSE-TREE: | | Scalar -> Integer -> Expr = '4_4'
!PARSE-TREE: | | | LiteralConstant -> IntLiteralConstant = '4'
!PARSE-TREE: | | Scalar -> Integer -> Expr = '5_4'
!PARSE-TREE: | | | LiteralConstant -> IntLiteralConstant = '5'
!PARSE-TREE: | | Scalar -> Integer -> Expr = '6_4'
!PARSE-TREE: | | | LiteralConstant -> IntLiteralConstant = '6'
!PARSE-TREE: | | Scalar -> Integer -> Expr = '7_4'
!PARSE-TREE: | | | LiteralConstant -> IntLiteralConstant = '7'
!PARSE-TREE: | Flags = {}
