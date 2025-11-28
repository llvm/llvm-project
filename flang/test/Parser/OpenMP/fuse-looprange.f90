! RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=60 %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=60  %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine openmp_fuse(x)

  integer, intent(inout)::x

!CHECK: !$omp fuse looprange
!$omp  fuse looprange(1,2)
!CHECK: do
  do x = 1, 100
  	call F1()
!CHECK: end do
  end do
!CHECK: do
  do x = 1, 100
  	call F1()
!CHECK: end do
  end do
!CHECK: do
  do x = 1, 100
  	call F1()
!CHECK: end do
  end do
!CHECK: !$omp end fuse
!$omp end fuse

!PARSE-TREE: OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE: OmpBeginLoopDirective
!PARSE-TREE: OmpDirectiveName -> llvm::omp::Directive = fuse
!PARSE-TREE: OmpClauseList -> OmpClause -> Looprange -> OmpLoopRangeClause
!PARSE-TREE: Scalar -> Integer -> Constant -> Expr = '1_4'
!PARSE-TREE: LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: Scalar -> Integer -> Constant -> Expr = '2_4'
!PARSE-TREE: LiteralConstant -> IntLiteralConstant = '2'

END subroutine openmp_fuse

