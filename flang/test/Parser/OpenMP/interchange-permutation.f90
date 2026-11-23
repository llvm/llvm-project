! RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=60 %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=60 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine openmp_interchange(x)

  integer :: x, y

!CHECK: !$omp interchange permutation(2_4,1_4)
!$omp interchange permutation(2,1)
!CHECK: do
  do x = 1, 100
  !CHECK: do
    do y = 1, 100
      call F1()
  !CHECK: end do
    end do
!CHECK: end do
  end do
!CHECK: !$omp end interchange
!$omp end interchange

!PARSE-TREE: OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE: OmpBeginDirective
!PARSE-TREE:  OmpDirectiveName -> llvm::omp::Directive = interchange
!PARSE-TREE:   OmpClauseList -> OmpClause -> Permutation -> Scalar -> Integer -> Constant -> Expr = '2_4'
!PARSE-TREE:     LiteralConstant -> IntLiteralConstant = '2'
!PARSE-TREE:   Scalar -> Integer -> Constant -> Expr = '1_4'
!PARSE-TREE:     LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE:     Flags = {}
!PARSE-TREE:   DoConstruct
!PARSE-TREE:   EndDoStmt
!PARSE-TREE: OmpEndDirective
!PARSE-TREE: OmpDirectiveName -> llvm::omp::Directive = interchange

END subroutine openmp_interchange
