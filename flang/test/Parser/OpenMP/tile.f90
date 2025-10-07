! RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=51 %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=51 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine openmp_tiles(x)

  integer, intent(inout)::x

!CHECK: !$omp tile sizes(2_4)
!$omp tile sizes(2)
!CHECK: do
  do x = 1, 100
  	call F1()
!CHECK: end do
  end do
!CHECK: !$omp end tile
!$omp end tile

!PARSE-TREE: OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE: OmpBeginLoopDirective
!PARSE-TREE:   OmpClauseList -> OmpClause -> Sizes -> Scalar -> Integer -> Expr = '2_4'
!PARSE-TREE:     LiteralConstant -> IntLiteralConstant = '2'
!PARSE-TREE:     Flags = None
!PARSE-TREE:   DoConstruct
!PARSE-TREE:   EndDoStmt
!PARSE-TREE: OmpEndLoopDirective
!PARSE-TREE: OmpDirectiveName -> llvm::omp::Directive = tile

END subroutine openmp_tiles
