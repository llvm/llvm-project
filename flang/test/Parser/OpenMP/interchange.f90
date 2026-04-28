! RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=60 %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=60 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine openmp_interchange(x)

  integer :: x, y

!CHECK: !$omp interchange
!$omp interchange
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
!PARSE-TREE:   DoConstruct
!PARSE-TREE:   EndDoStmt
!PARSE-TREE: OmpEndDirective
!PARSE-TREE: OmpDirectiveName -> llvm::omp::Directive = interchange

END subroutine openmp_interchange
