! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -fopenmp-version=51 -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: OpenMP ASSUMES construct
program p
  integer r
  r = 1
!$omp assumes no_parallelism
  print *,r
!$omp end assumes
end program p
