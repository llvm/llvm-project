!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=60 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=60 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine f00(x)
  integer :: x(10)
!$omp task threadset(omp_pool)
  x = x + 1
!$omp end task
end

!UNPARSE: SUBROUTINE f00 (x)
!UNPARSE:  INTEGER x(10_4)
!UNPARSE: !$OMP TASK THREADSET(OMP_POOL)
!UNPARSE:   x=x+1_4
!UNPARSE: !$OMP END TASK
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginDirective
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = task
!PARSE-TREE: | OmpClauseList -> OmpClause -> Threadset -> OmpThreadsetClause -> ThreadsetPolicy = Omp_Pool

subroutine f001(x)
  integer :: x(10)
!$omp task threadset(omp_team)
  x = x + 1
!$omp end task
end

!UNPARSE: SUBROUTINE f001 (x)
!UNPARSE:  INTEGER x(10_4)
!UNPARSE: !$OMP TASK THREADSET(OMP_TEAM)
!UNPARSE:   x=x+1_4
!UNPARSE: !$OMP END TASK
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginDirective
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = task
!PARSE-TREE: | OmpClauseList -> OmpClause -> Threadset -> OmpThreadsetClause -> ThreadsetPolicy = Omp_Team


subroutine f002(x)
  integer :: i
!$omp taskloop threadset(omp_team)
  do i = 1, 10
  end do
!$omp end taskloop
end

!UNPARSE: SUBROUTINE f002 (x)
!UNPARSE:  INTEGER i
!UNPARSE: !$OMP TASKLOOP THREADSET(OMP_TEAM)
!UNPARSE:    DO i=1_4,10_4
!UNPARSE:    END DO
!UNPARSE: !$OMP END TASK
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginLoopDirective
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = taskloop
!PARSE-TREE: | OmpClauseList -> OmpClause -> Threadset -> OmpThreadsetClause -> ThreadsetPolicy = Omp_Team

subroutine f003(x)
  integer :: i
!$omp taskloop threadset(omp_pool)
  do i = 1, 10
  end do
!$omp end taskloop
end

!UNPARSE: SUBROUTINE f003 (x)
!UNPARSE:  INTEGER i
!UNPARSE: !$OMP TASKLOOP THREADSET(OMP_POOL)
!UNPARSE:    DO i=1_4,10_4
!UNPARSE:    END DO
!UNPARSE: !$OMP END TASK
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginLoopDirective
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = taskloop
!PARSE-TREE: | OmpClauseList -> OmpClause -> Threadset -> OmpThreadsetClause -> ThreadsetPolicy = Omp_Pool
