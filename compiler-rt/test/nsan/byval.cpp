// RUN: %clangxx_nsan -O0 -g %s -o %t
// RUN: env NSAN_OPTIONS=halt_on_error=0:resume_after_warning=0 %run %t 2>&1 | FileCheck %s

// RUN: %clangxx_nsan -O2 -g %s -o %t
// RUN: env NSAN_OPTIONS=halt_on_error=0:resume_after_warning=0 %run %t 2>&1 | FileCheck %s

// A struct passed by value in memory (`byval`) is copied by the call into fresh
// stack memory. Check that the shadow of the caller's struct reaches the callee
// (so that accumulated error is still detected there), and that stale shadow
// left on the stack by earlier calls is not picked up.

// resume_after_warning=0 keeps the shadow after a warning. Otherwise, with
// optimization, the checked store of `s.x` in the caller would reset its shadow
// to the native value and the callee would see no error.

#include <cstdio>

struct S {
  double x, y, z;
};

__attribute__((noinline)) void Dirty(double v) {
  volatile double a[32];
  for (int i = 0; i < 32; i++)
    a[i] = v;
}

__attribute__((noinline)) double Use(S s) { return s.x; }

// The native result is 0.0 (1e17 + 1 rounds back to 1e17), but the shadow
// result is 1.0.
__attribute__((noinline)) double UseWithError(volatile double *one) {
  S s;
  s.x = (1e17 + *one) - 1e17;
  s.y = 0.0;
  s.z = 0.0;
  return Use(s);
}

__attribute__((noinline)) double UseWithoutError() {
  S s = {0.0, 0.0, 0.0};
  return Use(s);
}

int main() {
  volatile double one = 1.0;

  // No error in the argument: stale shadow from `Dirty` must not be reported.
  Dirty(4.0);
  double r1 = UseWithoutError();

  fprintf(stderr, "BEGIN-ERROR-CASE\n");
  // CHECK-NOT: WARNING
  // CHECK: BEGIN-ERROR-CASE
  // Error in the argument: it must be reported when the callee returns it. With
  // optimization the checked store in the caller may warn first.
  double r2 = UseWithError(&one);
  // CHECK: WARNING: NumericalStabilitySanitizer: inconsistent shadow results while checking return value
  // CHECK: double {{.*}}(native): dec: 0.0
  // CHECK: __float128 {{.*}}(shadow): dec: 1.0
  return r1 + r2 != 0.0;
}
