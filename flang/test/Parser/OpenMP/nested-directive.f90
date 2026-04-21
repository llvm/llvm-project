! RUN: %flang_fc1 -fdebug-unparse -fopenmp %s 2>&1 | FileCheck %s --match-full-lines

subroutine func
  implicit none
! CHECK: !$OMP NOTHING
  !$omp nothing !$omp Cannot nest directives inside directives; must be interpreted as a comment
end subroutine func
