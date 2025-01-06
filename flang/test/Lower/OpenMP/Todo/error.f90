! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -fopenmp-version=51 -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: OpenMPUtilityConstruct
program p
  integer, allocatable :: x
  !$omp error at(compilation) severity(warning) message("an error")
end program p
