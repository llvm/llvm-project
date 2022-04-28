! This test checks lowering of OpenACC routine Directive.

// RUN: not flang-new -fc1 -emit-fir -fopenacc %s 2>&1 | FileCheck %s

program main
  // CHECK: not yet implemented: OpenACC Routine construct not lowered yet!
  !$acc routine(sub) seq
contains
  subroutine sub(a)
    real :: a(:)
  end
end
