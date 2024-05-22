// RUN: %clang_cc1 -triple x86_64-linux -verify=norounding -Wno-unknown-pragmas %s
// RUN: %clang_cc1 -triple x86_64-linux -verify=rounding %s -frounding-math -Wno-unknown-pragmas
// RUN: %clang_cc1 -triple x86_64-linux -verify=rounding %s -frounding-math -fexperimental-new-constant-interpreter -Wno-unknown-pragmas
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


// nextUp(1.F) == 0x1.000002p0F
static_assert(1.0F + 0x0.000001p0F == 0x1.0p0F, "");

char Arr01[1 + (1.0F + 0x0.000001p0F > 1.0F)];
static_assert(sizeof(Arr01) == 1, "");

struct S1 {
  int : (1.0F + 0x0.000001p0F > 1.0F);
  int f;
};
static_assert(sizeof(S1) == sizeof(int), "");

#pragma STDC FENV_ROUND FE_UPWARD
static_assert(1.0F + 0x0.000001p0F == 0x1.000002p0F, "");

char Arr01u[1 + (1.0F + 0x0.000001p0F > 1.0F)];
static_assert(sizeof(Arr01u) == 2, "");

struct S1u {
  int : (1.0F + 0x0.000001p0F > 1.0F);
  int f;
};
static_assert(sizeof(S1u) > sizeof(int), "");

#pragma STDC FENV_ROUND FE_DOWNWARD
static_assert(1.0F + 0x0.000001p0F == 1.0F, "");

char Arr01d[1 + (1.0F + 0x0.000001p0F > 1.0F)];
static_assert(sizeof(Arr01d) == 1, "");

struct S1d {
  int : (1.0F + 0x0.000001p0F > 1.0F);
  int f;
};
static_assert(sizeof(S1d) == sizeof(int), "");

constexpr float incr_down(float k) {
  float x = k;
  ++x;
  return x;
}

// 0x1.0p23 = 8388608.0, inc(8388608.0) = 8388609.0
static_assert(incr_down(0x1.0p23F) == 0x1.000002p23F, "");
// 0x1.0p24 = 16777216.0, inc(16777216.0) = 16777217.0 -> round down -> 16777216.0
static_assert(incr_down(0x1.0p24F) == 0x1.0p24F, "");

#pragma STDC FENV_ROUND FE_UPWARD
constexpr float incr_up(float k) {
  float x = k;
  ++x;
  return x;
}
static_assert(incr_up(0x1.0p23F) == 0x1.000002p23F, "");
// 0x1.0p24 = 16777216.0, inc(16777216.0) = 16777217.0 -> round up -> 16777218.0
static_assert(incr_up(0x1.0p24F) == 0x1.000002p24F, "");
