! RUN: %flang_fc1 -fdebug-unparse-no-sema -fopenmp %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine openmp_parse_unroll(x)

  integer, intent(inout)::x

!CHECK: !$omp unroll partial(3)
!$omp  unroll partial(3)
!CHECK: do
  do x = 1, 100
  	call F1()
!CHECK: end do
  end do
!CHECK: !$omp end unroll
!$omp end unroll

!PARSE-TREE: OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE: OmpBeginLoopDirective
!PARSE-TREE: OmpLoopDirective -> llvm::omp::Directive = unroll
!PARSE-TREE: OmpClauseList -> OmpClause -> Partial -> Scalar -> Integer -> Constant -> Expr = '3_4'
!PARSE-TREE: LiteralConstant -> IntLiteralConstant = '3'

END subroutine openmp_parse_unroll
