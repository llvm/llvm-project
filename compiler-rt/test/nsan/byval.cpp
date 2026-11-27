// RUN: %clangxx_nsan -O0 -g %s -o %t
// RUN: env NSAN_OPTIONS=halt_on_error=0 %run %t 2>&1 | FileCheck %s

// RUN: %clangxx_nsan -O2 -g %s -o %t
// RUN: env NSAN_OPTIONS=halt_on_error=0 %run %t 2>&1 | FileCheck %s

// A struct passed by value in memory (`byval`) is copied by the call into fresh
// stack memory. Check that the shadow of the caller's struct reaches the callee,
// and that stale shadow left on the stack by earlier calls is not picked up.
//
// The error in the argument is below the warning threshold when it is passed
// (relative error ~1e-17), so the caller does not warn. The callee cancels the
// large offset and exposes it, which is only possible if it received the
// shadow of the argument.

#include <cstdio>

struct S {
  double x, y, z;
};

__attribute__((noinline)) void Dirty(double v) {
  volatile double a[32];
  for (int i = 0; i < 32; i++)
    a[i] = v;
}

// Cancels the large offset. If the shadow of `s` did not reach the callee, the
// shadow of `s.x` would be recomputed from the native value, and the
// callee's shadow result would be 0.0 as well.
__attribute__((noinline)) double Use(S s) { return s.x - 1e17; }

// Native: 1e17 + 1.0 rounds to 1e17. Shadow: 1e17 + 1.0. The relative error is
// ~1e-17, far below the warning threshold, so nothing is reported here, but the
// callee's subtraction exposes it (native 0.0, shadow 1.0).
__attribute__((noinline)) double UseWithError(volatile double *one) {
  S s;
  s.x = 1e17 + *one;
  s.y = 0.0;
  s.z = 0.0;
  return Use(s);
}

// No error in the argument, but the stack below it has stale shadow from Dirty.
__attribute__((noinline)) double UseWithoutError() {
  S s = {1e17, 0.0, 0.0};
  return Use(s);
}

volatile double sink;

int main() {
  volatile double one = 1.0;

  Dirty(4.0);
  sink = UseWithoutError();

  fprintf(stderr, "BEGIN-ERROR-CASE\n");
  // CHECK-NOT: WARNING
  // CHECK: BEGIN-ERROR-CASE
  sink = UseWithError(&one);
  // CHECK: WARNING: NumericalStabilitySanitizer: inconsistent shadow results while checking return value
  // CHECK: double {{.*}}(native): dec: 0.0
  // CHECK: __float128 {{.*}}(shadow): dec: 1.0
  // CHECK: #0 {{.*}} in Use(S)
  // The warning above is the only one.
  // CHECK-NOT: WARNING
  fprintf(stderr, "END\n");
  // CHECK: END
  return 0;
}
