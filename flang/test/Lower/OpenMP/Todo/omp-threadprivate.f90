! This test checks lowering of OpenMP threadprivate Directive.

// RUN: not flang-new -fc1 -emit-fir -fopenmp %s 2>&1 | FileCheck %s

program main
  integer, save :: x, y

// CHECK: not yet implemented: OpenMPThreadprivate
  !$omp threadprivate(x, y)
end
