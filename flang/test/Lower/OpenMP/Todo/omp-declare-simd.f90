! This test checks lowering of OpenMP declare simd Directive.

// RUN: not flang -fc1 -emit-fir -fopenmp %s 2>&1 | FileCheck %s

subroutine sub(x, y)
  real, intent(inout) :: x, y

  // CHECK: not yet implemented: OpenMPDeclareSimdConstruct
  !$omp declare simd(sub) aligned(x)
  x = 3.14 + y
end
