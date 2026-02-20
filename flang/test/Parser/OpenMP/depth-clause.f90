!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=61 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=61 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine f00
  integer :: i, j
  !$omp fuse depth(2)
  do i = 1, 10
    do j = 1, 10
    end do
  end do
  do i = 1, 10
    do j = 1, 10
    end do
  end do
  !$omp end fuse
end

!UNPARSE: SUBROUTINE f00
!UNPARSE:  INTEGER i, j
!UNPARSE: !$OMP FUSE DEPTH(2_4)
!UNPARSE:  DO i=1_4,10_4
!UNPARSE:   DO j=1_4,10_4
!UNPARSE:   END DO
!UNPARSE:  END DO
!UNPARSE:  DO i=1_4,10_4
!UNPARSE:   DO j=1_4,10_4
!UNPARSE:   END DO
!UNPARSE:  END DO
!UNPARSE: !$OMP END FUSE
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE: | OmpBeginLoopDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = fuse
!PARSE-TREE: | | OmpClauseList -> OmpClause -> Depth -> Scalar -> Integer -> Constant ->
!PARSE-TREE:  = '2_4'
!PARSE-TREE: | | | LiteralConstant -> IntLiteralConstant = '2'
!PARSE-TREE: | | Flags = {}
