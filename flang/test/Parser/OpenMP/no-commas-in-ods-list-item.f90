!RUN: not %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=60 %s 2>&1 | FileCheck %s

!The "11:" below is a line number. It is the line with the "otherwise" clause.
!CHECK-LABEL: Could not parse
!CHECK-LABEL: 11:{{[0-9]+}}: error:

subroutine f
  integer :: i, j

  !$omp metadirective &
  !$omp   otherwise(parallel do num_threads(2), collapse(2))
  do i = 1, 10
    do j = 1, 10
    end do
  end do
end
