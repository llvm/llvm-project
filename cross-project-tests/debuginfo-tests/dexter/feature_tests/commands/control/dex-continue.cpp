// Purpose:
//    Test \DexStepFunction usage with \DexContinue. Continuing out of `c`
//    should result in stepping resuming in `a` (check there's no issue when
//    `b` is inlined). Then continuing out of `a` should run on to `f` where
//    stepping resumes again. Stepping out of `f` into `main`, run free
//    again until the program exits.
//
// This command is only implemented for debuggers with DAP support.
// UNSUPPORTED: system-windows
//
// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run -v --binary %t -- %s 2>&1 | FileCheck --dump-input-context=999999999 %s

int g = 0;
int c(int) {
  ++g;
  ++g;
  ++g;
  ++g;
  ++g;
  return 0;
}

__attribute__((always_inline))
int b(int) {
  ++g;
  return c(g);
}

int a(int) {
  ++g;
  b(g);
  ++g;
  return g;
}

void f() {
  ++g;
}

int main() {
  int x = a(g);
  f();
  return x;
}

// DexStepFunction('c')
// DexContinue(from_line=17, to_line=19)
// DexContinue(from_line=20)
// DexStepFunction('a')
// DexContinue(from_line=33)
// DexStepFunction('f')

// CHECK:      ## BEGIN ##
// CHECK-NEXT: .   [0, "a(int)", "{{.*}}dex-continue.cpp", 31, 3, "StopReason.BREAKPOINT", "StepKind.FUNC", []]
// CHECK-NEXT: .   [1, "a(int)", "{{.*}}dex-continue.cpp", 32, 5, "StopReason.STEP", "StepKind.VERTICAL_FORWARD", []]
// CHECK-NEXT: .   .   .   [2, "c(int)", "{{.*}}dex-continue.cpp", 16, 3, "StopReason.BREAKPOINT", "StepKind.FUNC", []]
// CHECK-NEXT: .   .   .   [3, "c(int)", "{{.*}}dex-continue.cpp", 17, 3, "StopReason.BREAKPOINT", "StepKind.VERTICAL_FORWARD", []]
// CHECK-NEXT: .   .   .   [4, "c(int)", "{{.*}}dex-continue.cpp", 19, 3, "StopReason.BREAKPOINT", "StepKind.VERTICAL_FORWARD", []]
// CHECK-NEXT: .   .   .   [5, "c(int)", "{{.*}}dex-continue.cpp", 20, 3, "StopReason.BREAKPOINT", "StepKind.VERTICAL_FORWARD", []]
// CHECK-NEXT: .   [6, "a(int)", "{{.*}}dex-continue.cpp", 33, 3, "StopReason.BREAKPOINT", "StepKind.VERTICAL_FORWARD", []]
// CHECK-NEXT: .   [7, "f()", "{{.*}}dex-continue.cpp", 38, 3, "StopReason.BREAKPOINT", "StepKind.VERTICAL_FORWARD", []]
// CHECK-NEXT: .   [8, "f()", "{{.*}}dex-continue.cpp", 39, 1, "StopReason.STEP", "StepKind.VERTICAL_FORWARD", []]
// CHECK-NEXT: ## END (9 steps) ##
