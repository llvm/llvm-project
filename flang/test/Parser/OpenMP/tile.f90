! RUN: %flang_fc1 -fdebug-unparse -fopenmp %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine openmp_tiles(x)

  integer, intent(inout)::x

!CHECK: !$omp tile
!$omp  tile
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

END subroutine openmp_tiles

