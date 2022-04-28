! This test checks lowering of OpenACC declare Directive.

// RUN: not flang-new -fc1 -emit-fir -fopenacc %s 2>&1 | FileCheck %s

program main
  real, dimension(10) :: aa, bb

  // CHECK: not yet implemented: OpenACC Standalone Declarative construct
  !$acc declare present(aa, bb)
end
