! Test lowering of of expressions as values
! RUN: %not_todo_cmd bbc -emit-fir -hlfir -o - %s 2>&1 | FileCheck %s

subroutine foo()
  ! CHECK: not yet implemented: lower expr to HLFIR value
  print *, 42 
end subroutine
