!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=60 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=60 %s | FileCheck --check-prefix="PARSE-TREE" %s

!UNPARSE: SUBROUTINE teams_workdistribute
!UNPARSE:  USE :: iso_fortran_env
!UNPARSE:  REAL(KIND=4_4) a
!UNPARSE:  REAL(KIND=4_4), DIMENSION(10_4) :: x
!UNPARSE:  REAL(KIND=4_4), DIMENSION(10_4) :: y
!UNPARSE: !$OMP TEAMS WORKDISTRIBUTE
!UNPARSE:   y=a*x+y
!UNPARSE: !$OMP END TEAMS WORKDISTRIBUTE
!UNPARSE: END SUBROUTINE teams_workdistribute

!PARSE-TREE: | | | OmpBeginDirective
!PARSE-TREE: | | | | OmpDirectiveName -> llvm::omp::Directive = teams workdistribute
!PARSE-TREE: | | | OmpEndDirective
!PARSE-TREE: | | | | OmpDirectiveName -> llvm::omp::Directive = teams workdistribute

subroutine teams_workdistribute()
  use iso_fortran_env
  real(kind=real32) :: a
  real(kind=real32), dimension(10) :: x
  real(kind=real32), dimension(10) :: y
  !$omp teams workdistribute
  y = a * x + y
  !$omp end teams workdistribute
end subroutine teams_workdistribute
