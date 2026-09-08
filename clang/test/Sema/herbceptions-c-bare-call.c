// RUN: not %clang_cc1 -fherbceptions -fsyntax-only %s 2>&1 | FileCheck %s

int bar(int x) return_failure{int} { return x; }

// In C, calling a return_failure{E} function without try() or catch return_failure() is a
// compile-time error.
// CHECK: error: calling function with 'return_failure{...}' specifier requires 'try()' or 'catch return_failure()' wrapper
// CHECK-NOT: error: expected function body
int foo(int x) {
  return bar(x);
}
