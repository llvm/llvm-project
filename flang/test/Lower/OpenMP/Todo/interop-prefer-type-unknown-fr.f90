! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -fopenmp-version=60 -o - %s 2>&1 | FileCheck %s

! A prefer_type foreign-runtime identifier string that is not one of the
! standard OpenMP names cannot be mapped to a runtime id.

! CHECK: not yet implemented: unknown foreign-runtime identifier in prefer_type

subroutine interop_prefer_type_unknown_fr(obj)
  integer(8) :: obj
  !$omp interop init(prefer_type("no_such_runtime"), target: obj)
end subroutine
