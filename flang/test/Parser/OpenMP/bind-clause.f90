!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=50 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=50 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine f00
  !$omp loop bind(parallel)
  do i = 1, 10
    continue
  enddo
  !$omp end loop
end

!UNPARSE: SUBROUTINE f00
!UNPARSE: !$OMP LOOP  BIND(PARALLEL)
!UNPARSE:  DO i=1_4,10_4
!UNPARSE:   CONTINUE
!UNPARSE:  END DO
!UNPARSE: !$OMP END LOOP
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE: | OmpBeginLoopDirective
!PARSE-TREE: | | OmpLoopDirective -> llvm::omp::Directive = loop
!PARSE-TREE: | | OmpClauseList -> OmpClause -> Bind -> OmpBindClause -> Binding = Parallel
!PARSE-TREE: | DoConstruct

