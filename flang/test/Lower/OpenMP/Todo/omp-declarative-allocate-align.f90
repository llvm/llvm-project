! This test checks lowering of OpenMP allocate Directive with align clause.

// RUN: not flang -fc1 -emit-fir -fopenmp %s 2>&1 | FileCheck %s

program main
  integer :: x

  // CHECK: not yet implemented: OpenMPDeclarativeAllocate
  !$omp allocate(x) align(32)
end
