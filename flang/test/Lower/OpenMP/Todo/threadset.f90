! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -fopenmp-version=60 -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: THREADSET clause is not implemented yet

subroutine f00(x)
  integer :: x(10)
  !$omp task threadset(omp_pool)
  x = x + 1
  !$omp end task
end
