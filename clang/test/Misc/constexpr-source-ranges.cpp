// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-print-source-range-info -fcxx-exceptions %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -fsyntax-only -fexperimental-new-constant-interpreter -fdiagnostics-print-source-range-info -fcxx-exceptions %s 2>&1 | FileCheck %s

constexpr int f() {
  throw 1;
  return 0;
}

// CHECK: constexpr-source-ranges.cpp:5:3:{5:3-5:10}


constexpr int I = 12;
constexpr const int *P = &I;
constexpr long L = (long)P;
// CHECK: constexpr-source-ranges.cpp:14:20:{14:20-14:27}

constexpr int zero() {
  return 0;
}
constexpr int divByZero() {
  return 1 / zero();
}
static_assert(divByZero() == 0, "");
/// We see this twice. Once from sema and once when
/// evaluating the static_assert above.
// CHECK: constexpr-source-ranges.cpp:23:15:{23:15-23:31}
// CHECK: constexpr-source-ranges.cpp:21:12:{21:14-21:20}
