! This test checks lowering of OpenMP declare target Directive.

// RUN: not flang-new -fc1 -emit-fir -fopenmp %s 2>&1 | FileCheck %s

module mod1
contains
  subroutine sub()
    integer :: x, y
    // CHECK: not yet implemented: OpenMPDeclareTargetConstruct
    !$omp declare target
  end
end module
