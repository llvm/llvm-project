! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -fopenmp-version=60 -o - %s 2>&1 | FileCheck %s

! Only default-kind CHARACTER foreign-runtime identifier strings are mapped to
! a runtime id; other kinds are treated as non-constant values.

! CHECK: not yet implemented: non-constant prefer_type value in interop init

subroutine interop_prefer_type_char_kind4(obj)
  integer(8) :: obj
  !$omp interop init(prefer_type(4_"cuda"), targetsync: obj)
end subroutine
