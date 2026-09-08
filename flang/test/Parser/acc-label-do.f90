! RUN: %flang_fc1 -fopenacc -fdebug-unparse %s | FileCheck --ignore-case --check-prefix=UNPARSE %s
! RUN: %flang_fc1 -fopenacc -fdebug-dump-parse-tree %s | FileCheck --check-prefix=PARSE-TREE %s

! Combined constructs associated with a non-block (labeled) DO loop must still
! accept an optional end directive that follows the terminating labeled
! statement.

subroutine s
  integer :: i, n
!$acc parallel loop
  do 10 i = 1, n
10 continue
!$acc end parallel
end

!UNPARSE: SUBROUTINE s
!UNPARSE:  INTEGER i, n
!UNPARSE: !$ACC PARALLEL LOOP
!UNPARSE:  DO i=1_4,n
!UNPARSE:   10 CONTINUE
!UNPARSE:  END DO
!UNPARSE: !$ACC END PARALLEL LOOP
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OpenACCCombinedConstruct
!PARSE-TREE: | AccBeginCombinedDirective
!PARSE-TREE: | | AccCombinedDirective -> llvm::acc::Directive = parallel loop
!PARSE-TREE: | DoConstruct
!PARSE-TREE: | | NonLabelDoStmt
!PARSE-TREE: | | Block
!PARSE-TREE: | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> ContinueStmt
!PARSE-TREE: | | EndDoStmt ->
!PARSE-TREE: | AccEndCombinedDirective -> AccCombinedDirective -> llvm::acc::Directive = parallel loop

! Nested labeled DO loops that share a terminating label, with an OpenACC LOOP
! construct associated with the inner loop.  The shared label must still close
! both loops.

subroutine nested_shared_label(a, n)
  integer :: i, j, n
  real :: a(n)
!$acc kernels
!$acc loop independent
  do 10 j = 1, n
!$acc loop gang vector
    do 10 i = 1, n
10     a(i) = 0.
!$acc end kernels
end

! Same, with three levels of nesting: the label of the innermost loop is
! reached through two OpenACC LOOP constructs.

subroutine triple_shared_label(a, c, n, k)
  integer :: i, j, m, n, k
  real :: a(n), c(n,k)
!$acc kernels
!$acc loop independent
  do 250 m = 1, n
!$acc loop independent
    do 250 j = 1, k
!$acc loop gang vector
      do 250 i = 1, n
250     a(i) = a(i) + c(i,j)
!$acc end kernels
end

!UNPARSE: SUBROUTINE nested_shared_label (a, n)
!UNPARSE: !$ACC KERNELS
!UNPARSE: !$ACC LOOP INDEPENDENT
!UNPARSE:  DO j=1_4,n
!UNPARSE: !$ACC LOOP GANG VECTOR
!UNPARSE:   DO i=1_4,n
!UNPARSE:    10
!UNPARSE:   END DO
!UNPARSE:  END DO
!UNPARSE: !$ACC END KERNELS
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OpenACCBlockConstruct
!PARSE-TREE: | AccBeginBlockDirective
!PARSE-TREE: | | AccBlockDirective -> llvm::acc::Directive = kernels
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> OpenACCConstruct -> OpenACCLoopConstruct
!PARSE-TREE: | | | AccBeginLoopDirective
!PARSE-TREE: | | | | AccLoopDirective -> llvm::acc::Directive = loop
!PARSE-TREE: | | | | AccClauseList -> AccClause -> Independent
!PARSE-TREE: | | | DoConstruct
!PARSE-TREE: | | | | NonLabelDoStmt
!PARSE-TREE: | | | | Block
!PARSE-TREE: | | | | | ExecutionPartConstruct -> ExecutableConstruct -> OpenACCConstruct -> OpenACCLoopConstruct
!PARSE-TREE: | | | | | | AccBeginLoopDirective
!PARSE-TREE: | | | | | | | AccLoopDirective -> llvm::acc::Directive = loop
!PARSE-TREE: | | | | | | DoConstruct
!PARSE-TREE: | | | | | | | NonLabelDoStmt
!PARSE-TREE: | | | | | | | Block
!PARSE-TREE: | | | | | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt
!PARSE-TREE: | | | | | | | EndDoStmt ->
!PARSE-TREE: | | | | EndDoStmt ->
!PARSE-TREE: | AccEndBlockDirective -> AccBlockDirective -> llvm::acc::Directive = kernels

!UNPARSE: SUBROUTINE triple_shared_label (a, c, n, k)
!UNPARSE: !$ACC KERNELS
!UNPARSE: !$ACC LOOP INDEPENDENT
!UNPARSE:  DO m=1_4,n
!UNPARSE: !$ACC LOOP INDEPENDENT
!UNPARSE:   DO j=1_4,k
!UNPARSE: !$ACC LOOP GANG VECTOR
!UNPARSE:    DO i=1_4,n
!UNPARSE:     250
!UNPARSE:    END DO
!UNPARSE:   END DO
!UNPARSE:  END DO
!UNPARSE: !$ACC END KERNELS
!UNPARSE: END SUBROUTINE
