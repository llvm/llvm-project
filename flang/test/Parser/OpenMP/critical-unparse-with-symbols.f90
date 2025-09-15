!RUN: %flang_fc1 -fdebug-unparse-with-symbols -fopenmp -fopenmp-version=50 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s

subroutine f
  implicit none
  integer :: x
  !$omp critical(c)
  x = 0
  !$omp end critical(c)
end

!UNPARSE: !DEF: /f (Subroutine) Subprogram
!UNPARSE: subroutine f
!UNPARSE:  implicit none
!UNPARSE:  !DEF: /f/x ObjectEntity INTEGER(4)
!UNPARSE:  integer x
!UNPARSE: !$omp critical(c)
!UNPARSE:  !REF: /f/x
!UNPARSE:  x = 0
!UNPARSE: !$omp end critical(c)
!UNPARSE: end subroutine

