! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -fopenmp-version=60 -o - %s 2>&1 | FileCheck %s

! A prefer_type preference-selector list (the OpenMP 6.0 brace form) is valid
! syntax but not yet lowered.

! CHECK: not yet implemented: prefer_type with preference-selector lists in interop init clause

subroutine interop_prefer_type_selector_list(obj)
  integer(8) :: obj
  !$omp interop init(prefer_type({fr("cuda")}), target: obj)
end subroutine
