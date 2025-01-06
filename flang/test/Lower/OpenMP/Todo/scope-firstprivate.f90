! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -o - %s -fopenmp-version=52 2>&1 | FileCheck %s

! CHECK: not yet implemented: Scope construct
program omp_scope
  integer i
  i = 10

  !$omp scope firstprivate(x)
  print *, "omp scope", i
  !$omp end scope

end program omp_scope
