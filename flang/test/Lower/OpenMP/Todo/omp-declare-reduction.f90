! This test checks lowering of OpenMP declare reduction Directive.

// RUN: not flang-new -fc1 -emit-fir -fopenmp %s 2>&1 | FileCheck %s

subroutine declare_red()
  integer :: my_var
  // CHECK: not yet implemented: OpenMPDeclareReductionConstruct
  !$omp declare reduction (my_red : integer : omp_out = omp_in) initializer (omp_priv = 0)
  my_var = 0
end subroutine declare_red
