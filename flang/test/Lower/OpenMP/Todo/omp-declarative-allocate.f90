! This test checks lowering of OpenMP allocate Directive.

// RUN: not flang -fc1 -emit-fir -fopenmp %s 2>&1 | FileCheck %s

program main
  integer :: x, y

  // CHECK: OpenMP directive ALLOCATE has been deprecated, please use ALLOCATORS instead.
  // CHECK: not yet implemented: OpenMPDeclarativeAllocate
  !$omp allocate(x, y)
end
