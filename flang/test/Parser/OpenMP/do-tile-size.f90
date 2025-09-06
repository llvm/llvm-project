! RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=51 %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=51 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine openmp_do_tiles(x)

  integer, intent(inout)::x


!CHECK: !$omp do
!CHECK: !$omp tile sizes
!$omp do
!$omp  tile sizes(2)
!CHECK: do
  do x = 1, 100
     call F1()
!CHECK: end do
  end do
!CHECK: !$omp end tile
!$omp end tile
!$omp end do

!PARSE-TREE:| | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE:| | | OmpBeginLoopDirective
!PARSE-TREE:| | | OpenMPLoopConstruct
!PARSE-TREE:| | | | OmpBeginLoopDirective
!PARSE-TREE:| | | | | OmpLoopDirective -> llvm::omp::Directive = tile
!PARSE-TREE:| | | | | OmpClauseList -> OmpClause -> Sizes -> Scalar -> Integer -> Expr = '2_4'
!PARSE-TREE: | | | | DoConstruct
END subroutine openmp_do_tiles
