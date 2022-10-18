! Test lowering of of expressions as address
! RUN: %not_todo_cmd bbc -emit-fir -hlfir -o - %s 2>&1 | FileCheck %s

subroutine foo(x)
  integer :: x
  ! CHECK: not yet implemented: generate fir.declare when lowering symbol
  read (*,*) x
end subroutine
