!RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 -o - %s 2>&1 | FileCheck %s

!CHECK: not yet implemented: Reserved locators are not supported yet

subroutine f
  ! This is a wrong use of OMP_ALL_MEMORY, but at the moment the clauses that
  ! legally allow this locator aren't accepting it yet in flang.
  !$omp target map(omp_all_memory)
  !$omp end target
end

