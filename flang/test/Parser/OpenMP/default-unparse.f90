! RUN: %flang_fc1 -fdebug-unparse-no-sema -fopenmp %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema -fopenmp %s | FileCheck --check-prefix="PARSE-TREE" %s
program main
!CHECK-LABEL: program main
  implicit none
    integer, target :: s = 0
    integer, pointer :: p
    integer :: arr(10)
    integer, allocatable :: al(:)

    allocate(al(10))

!CHECK: !$omp parallel default(none)
  !$omp parallel default(none)
    s = 1
!CHECK: !$omp end parallel
  !$omp end parallel

!PARSE-TREE:      OmpBeginDirective
!PARSE-TREE:        OmpDirectiveName -> llvm::omp::Directive = parallel
!PARSE-TREE:        OmpClauseList -> OmpClause -> Default -> OmpDefaultClause
!PARSE-TREE:          DataSharingAttribute = None

!CHECK: !$omp parallel default(private:scalar)
  !$omp parallel default(private:scalar)
    s = 1
!CHECK: !$omp end parallel
  !$omp end parallel

!PARSE-TREE:      OmpBeginDirective
!PARSE-TREE:        OmpDirectiveName -> llvm::omp::Directive = parallel
!PARSE-TREE:        OmpClauseList -> OmpClause -> Default -> OmpDefaultClause
!PARSE-TREE:          DataSharingAttribute = Private
!PARSE-TREE:          Modifier -> OmpVariableCategory -> Value = Scalar

!CHECK: !$omp parallel default(firstprivate:aggregate)
  !$omp parallel default(firstprivate:aggregate)
    arr = 1
!CHECK: !$omp end parallel
  !$omp end parallel

!PARSE-TREE:      OmpBeginDirective
!PARSE-TREE:        OmpDirectiveName -> llvm::omp::Directive = parallel
!PARSE-TREE:        OmpClauseList -> OmpClause -> Default -> OmpDefaultClause
!PARSE-TREE:          DataSharingAttribute = Firstprivate
!PARSE-TREE:          Modifier -> OmpVariableCategory -> Value = Aggregate

!CHECK: !$omp parallel default(shared:pointer)
  !$omp parallel default(shared:pointer)
    p=>s
    p = 1
!CHECK: !$omp end parallel
  !$omp end parallel

!PARSE-TREE:      OmpBeginDirective
!PARSE-TREE:        OmpDirectiveName -> llvm::omp::Directive = parallel
!PARSE-TREE:        OmpClauseList -> OmpClause -> Default -> OmpDefaultClause
!PARSE-TREE:          DataSharingAttribute = Shared
!PARSE-TREE:          Modifier -> OmpVariableCategory -> Value = Pointer

!CHECK: !$omp parallel default(private:allocatable)
  !$omp parallel default(private:allocatable)
    al = 1
!CHECK: !$omp end parallel
  !$omp end parallel

!PARSE-TREE:      OmpBeginDirective
!PARSE-TREE:        OmpDirectiveName -> llvm::omp::Directive = parallel
!PARSE-TREE:        OmpClauseList -> OmpClause -> Default -> OmpDefaultClause
!PARSE-TREE:          DataSharingAttribute = Private
!PARSE-TREE:          Modifier -> OmpVariableCategory -> Value = Allocatable

end program main
!CHECK-LABEL: end program main
