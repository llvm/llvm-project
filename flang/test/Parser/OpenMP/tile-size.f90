! RUN: %flang_fc1 -fdebug-unparse -fopenmp %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine openmp_tiles(x)

  integer, intent(inout)::x

!CHECK: !$omp tile sizes
!$omp  tile sizes(2)
!CHECK: do
  do x = 1, 100
  	call F1()
!CHECK: end do
  end do
!CHECK: !$omp end tile
!$omp end tile


!PARSE-TREE: OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE: OmpBeginLoopDirective
!PARSE-TREE: OmpLoopDirective -> llvm::omp::Directive = tile
!PARSE-TREE: OmpClauseList -> OmpClause -> Sizes -> Scalar -> Integer -> Expr = '2_4'
END subroutine openmp_tiles
