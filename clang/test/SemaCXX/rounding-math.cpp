// RUN: %clang_cc1 -triple x86_64-linux -verify=norounding %s
// RUN: %clang_cc1 -triple x86_64-linux -verify=rounding %s -frounding-math
// rounding-no-diagnostics

#define fold(x) (__builtin_constant_p(x) ? (x) : (x))

constexpr double a = 1.0 / 3.0;

constexpr int f(int n) { return int(n * (1.0 / 3.0)); }

using T = int[f(3)];
using T = int[1];

enum Enum { enum_a = f(3) };

struct Bitfield {
  unsigned int n : 1;
  unsigned int m : f(3);
};

void f(Bitfield &b) {
  b.n = int(6 * (1.0 / 3.0)); // norounding-warning {{changes value from 2 to 0}}
}

const int k = 3 * (1.0 / 3.0);
static_assert(k == 1, "");

void g() {
  // FIXME: Constant-evaluating this initializer is surprising, and violates
  // the recommended practice in C++ [expr.const]p12:
  //
  //   Implementations should provide consistent results of floating-point
  //   evaluations, irrespective of whether the evaluation is performed during
  //   translation or during program execution.
  const int k = 3 * (1.0 / 3.0);
  static_assert(k == 1, "");
}

int *h() {
  return new int[int(-3 * (1.0 / 3.0))]; // norounding-error {{too large}}
}
