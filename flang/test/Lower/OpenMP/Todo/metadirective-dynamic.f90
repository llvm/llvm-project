! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: dynamic user condition in METADIRECTIVE

subroutine test_dynamic_user_condition(flag)
  logical, intent(in) :: flag
  !$omp metadirective &
  !$omp & when(user={condition(flag)}: taskyield) &
  !$omp & default(nothing)
end subroutine
