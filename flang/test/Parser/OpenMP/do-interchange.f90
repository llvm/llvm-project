! RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=60 %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=60 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine openmp_do_interchange(x)

  integer :: x, y

!CHECK: !$omp do
!CHECK: !$omp interchange permutation
!$omp do
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
!$omp end do

!PARSE-TREE:| | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE:| | | OmpBeginDirective
!PARSE-TREE:| | | Block
!PARSE-TREE:| | | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE:| | | | | OmpBeginDirective
!PARSE-TREE:| | | | | | OmpDirectiveName -> llvm::omp::Directive = interchange
!PARSE-TREE:| | | | | Block
!PARSE-TREE:| | | | | | ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct

END subroutine openmp_do_interchange
