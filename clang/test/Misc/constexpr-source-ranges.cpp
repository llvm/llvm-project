// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-print-source-range-info -fcxx-exceptions %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -fsyntax-only -fexperimental-new-constant-interpreter -fdiagnostics-print-source-range-info -fcxx-exceptions %s 2>&1 | FileCheck %s

constexpr int f() {
  throw 1;
  return 0;
}
// CHECK: :[[@LINE-3]]:3:{[[@LINE-3]]:3-[[@LINE-3]]:10}

constexpr int I = 12;
constexpr const int *P = &I;
constexpr long L = (long)P;
// CHECK: :[[@LINE-1]]:20:{[[@LINE-1]]:20-[[@LINE-1]]:27}

constexpr int zero() {
  return 0;
}
constexpr int divByZero() {
  return 1 / zero();
}
static_assert(divByZero() == 0, "");
/// We see this twice. Once from sema and once when
/// evaluating the static_assert above.
// CHECK: :[[@LINE-5]]:12:{[[@LINE-5]]:14-[[@LINE-5]]:20}
// CHECK: :[[@LINE-4]]:15:{[[@LINE-4]]:15-[[@LINE-4]]:31}

constexpr int div(bool a, bool b) {
  return 1 / (int)b;
}
constexpr int ints(int a, int b, int c, int d) {
  return 1;
}
static_assert(ints(1, div(true, false), 2, div(false, true)) == 1, "");
// CHECK: :[[@LINE-1]]:23:{[[@LINE-1]]:23-[[@LINE-1]]:39}

namespace overflow {
// CHECK:      :{[[@LINE+1]]:9-[[@LINE+1]]:29}:
int x = -1 + __INT_MAX__ + 2 + 3;
// CHECK:      :{[[@LINE+1]]:9-[[@LINE+1]]:19}:
int a = -(1 << 31) + 1;
}


constexpr int uninit() {
  int aaa;
  // CHECK: :{[[@LINE+1]]:10-[[@LINE+1]]:13}:
  return aaa;
}
static_assert(uninit() == 0, "");


constexpr void neverValid() { throw; }
// CHECK: :{[[@LINE-1]]:16-[[@LINE-1]]:26}:

struct B1  {};
struct C  {};
constexpr C c;
// CHECK: :[[@LINE+1]]:15:{[[@LINE+1]]:15-[[@LINE+1]]:18}
constexpr int foo() {
  auto p = (B1&)c;
  return 1;
}
// CHECK: :[[@LINE-3]]:12:{[[@LINE-3]]:12-[[@LINE-3]]:18}
