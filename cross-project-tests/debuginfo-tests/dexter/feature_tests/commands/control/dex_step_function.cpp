// Purpose:
//    \DexStepFunction smoke test. Only steps in a and c should be logged.
//
// This command is only implemented for debuggers with DAP support.
// UNSUPPORTED: system-windows
//
// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run -v --binary %t -- %s 2>&1 | FileCheck %s

int g = 0;
int c(int) {
  ++g;
  return 0;
}

int b(int) {
  ++g;
  return c(g);
}

int a(int) {
  ++g;
  return b(g);
}

int main() {
  return a(g);
}

// DexStepFunction('a')
// DexStepFunction('c')

// CHECK:      ## BEGIN ##
// CHECK-NEXT:.   [0, "a(int)", "{{.*}}dex_step_function.cpp", 22, 3, "StopReason.BREAKPOINT", "StepKind.FUNC", []]
// CHECK-NEXT:.   [1, "a(int)", "{{.*}}dex_step_function.cpp", 23, 12, "StopReason.STEP", "StepKind.VERTICAL_FORWARD", []]
// CHECK-NEXT:.   .   .   [2, "c(int)", "{{.*}}dex_step_function.cpp", 12, 3, "StopReason.BREAKPOINT", "StepKind.FUNC", []]
// CHECK-NEXT:.   .   .   [3, "c(int)", "{{.*}}dex_step_function.cpp", 13, 3, "StopReason.STEP", "StepKind.VERTICAL_FORWARD", []]
// CHECK-NEXT:.   [4, "a(int)", "{{.*}}dex_step_function.cpp", 23, 3, "StopReason.STEP", "StepKind.HORIZONTAL_BACKWARD", []]
// CHECK-NEXT: ## END (5 steps) ##
