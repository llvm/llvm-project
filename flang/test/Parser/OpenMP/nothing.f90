!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=51 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=51 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine f00
  !$omp nothing
end

!UNPARSE: SUBROUTINE f00
!UNPARSE:  !$OMP NOTHING
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPart -> Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPUtilityConstruct -> OmpNothingDirective
