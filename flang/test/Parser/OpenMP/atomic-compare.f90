! RUN: not %flang_fc1  -fopenmp-version=51 -fopenmp %s 2>&1 | FileCheck %s
! OpenMP version for documentation purposes only - it isn't used until Sema.
! This is testing for Parser errors that bail out before Sema. 
program main
   implicit none
   integer :: i, j = 10
   logical :: r

  !CHECK: error: expected OpenMP construct
  !$omp atomic compare write
  r =  i .eq. j + 1

  !CHECK: error: expected end of line
  !$omp atomic compare num_threads(4)
  r = i .eq. j
end program main
