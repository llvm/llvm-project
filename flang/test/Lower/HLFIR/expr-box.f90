! Test lowering of of expressions as fir.box
! RUN: %not_todo_cmd bbc -emit-fir -hlfir -o - %s 2>&1 | FileCheck %s

subroutine foo(x)
  integer :: x(:)
  ! CHECK: not yet implemented: lower expr to HLFIR box
  print *, x 
end subroutine
