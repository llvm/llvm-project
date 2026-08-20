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
