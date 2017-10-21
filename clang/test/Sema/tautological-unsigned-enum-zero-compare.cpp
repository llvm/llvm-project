// RUN: %clang_cc1 -std=c++11 -triple=x86_64-pc-linux-gnu -fsyntax-only -DUNSIGNED -verify %s
// RUN: %clang_cc1 -std=c++11 -triple=x86_64-pc-win32 -fsyntax-only -DSIGNED -verify %s
// RUN: %clang_cc1 -std=c++11 -triple=x86_64-pc-win32 -fsyntax-only -DSILENCE -Wno-tautological-unsigned-enum-zero-compare -verify %s

// Okay, this is where it gets complicated.
// Then default enum sigdness is target-specific.
// On windows, it is signed by default. We do not want to warn in that case.

int main() {
  enum A { A_foo = 0, A_bar, };
  enum A a;

  enum B : unsigned { B_foo = 0, B_bar, };
  enum B b;

  enum C : signed { C_foo = 0, C_bar, };
  enum C c;

#ifdef UNSIGNED
  if (a < 0) // expected-warning {{comparison of unsigned enum expression < 0 is always false}}
    return 0;
  if (0 >= a)
    return 0;
  if (a > 0)
    return 0;
  if (0 <= a) // expected-warning {{comparison of 0 <= unsigned enum expression is always true}}
    return 0;
  if (a <= 0)
    return 0;
  if (0 > a) // expected-warning {{comparison of 0 > unsigned enum expression is always false}}
    return 0;
  if (a >= 0) // expected-warning {{comparison of unsigned enum expression >= 0 is always true}}
    return 0;
  if (0 < a)
    return 0;

  if (a < 0U) // expected-warning {{comparison of unsigned enum expression < 0 is always false}}
    return 0;
  if (0U >= a)
    return 0;
  if (a > 0U)
    return 0;
  if (0U <= a) // expected-warning {{comparison of 0 <= unsigned enum expression is always true}}
    return 0;
  if (a <= 0U)
    return 0;
  if (0U > a) // expected-warning {{comparison of 0 > unsigned enum expression is always false}}
    return 0;
  if (a >= 0U) // expected-warning {{comparison of unsigned enum expression >= 0 is always true}}
    return 0;
  if (0U < a)
    return 0;

  if (b < 0) // expected-warning {{comparison of unsigned enum expression < 0 is always false}}
    return 0;
  if (0 >= b)
    return 0;
  if (b > 0)
    return 0;
  if (0 <= b) // expected-warning {{comparison of 0 <= unsigned enum expression is always true}}
    return 0;
  if (b <= 0)
    return 0;
  if (0 > b) // expected-warning {{comparison of 0 > unsigned enum expression is always false}}
    return 0;
  if (b >= 0) // expected-warning {{comparison of unsigned enum expression >= 0 is always true}}
    return 0;
  if (0 < b)
    return 0;

  if (b < 0U) // expected-warning {{comparison of unsigned enum expression < 0 is always false}}
    return 0;
  if (0U >= b)
    return 0;
  if (b > 0U)
    return 0;
  if (0U <= b) // expected-warning {{comparison of 0 <= unsigned enum expression is always true}}
    return 0;
  if (b <= 0U)
    return 0;
  if (0U > b) // expected-warning {{comparison of 0 > unsigned enum expression is always false}}
    return 0;
  if (b >= 0U) // expected-warning {{comparison of unsigned enum expression >= 0 is always true}}
    return 0;
  if (0U < b)
    return 0;

  if (c < 0)
    return 0;
  if (0 >= c) // expected-warning {{comparison 0 >= 'enum C' is always true}}
    return 0;
  if (c > 0) // expected-warning {{comparison 'enum C' > 0 is always false}}
    return 0;
  if (0 <= c)
    return 0;
  if (c <= 0) // expected-warning {{comparison 'enum C' <= 0 is always true}}
    return 0;
  if (0 > c)
    return 0;
  if (c >= 0)
    return 0;
  if (0 < c) // expected-warning {{0 < 'enum C' is always false}}
    return 0;

  if (c < 0U) // expected-warning {{comparison of unsigned enum expression < 0 is always false}}
    return 0;
  if (0U >= c)
    return 0;
  if (c > 0U)
    return 0;
  if (0U <= c) // expected-warning {{comparison of 0 <= unsigned enum expression is always true}}
    return 0;
  if (c <= 0U)
    return 0;
  if (0U > c) // expected-warning {{comparison of 0 > unsigned enum expression is always false}}
    return 0;
  if (c >= 0U) // expected-warning {{comparison of unsigned enum expression >= 0 is always true}}
    return 0;
  if (0U < c)
    return 0;
#elif defined(SIGNED)
  if (a < 0)
    return 0;
  if (0 >= a) // expected-warning {{comparison 0 >= 'enum A' is always true}}
    return 0;
  if (a > 0) // expected-warning {{comparison 'enum A' > 0 is always false}}
    return 0;
  if (0 <= a)
    return 0;
  if (a <= 0) // expected-warning {{comparison 'enum A' <= 0 is always true}}
    return 0;
  if (0 > a)
    return 0;
  if (a >= 0)
    return 0;
  if (0 < a) // expected-warning {{comparison 0 < 'enum A' is always false}}
    return 0;

  if (a < 0U) // expected-warning {{comparison of unsigned enum expression < 0 is always false}}
    return 0;
  if (0U >= a)
    return 0;
  if (a > 0U)
    return 0;
  if (0U <= a) // expected-warning {{comparison of 0 <= unsigned enum expression is always true}}
    return 0;
  if (a <= 0U)
    return 0;
  if (0U > a) // expected-warning {{comparison of 0 > unsigned enum expression is always false}}
    return 0;
  if (a >= 0U) // expected-warning {{comparison of unsigned enum expression >= 0 is always true}}
    return 0;
  if (0U < a)
    return 0;

  if (b < 0) // expected-warning {{comparison of unsigned enum expression < 0 is always false}}
    return 0;
  if (0 >= b)
    return 0;
  if (b > 0)
    return 0;
  if (0 <= b) // expected-warning {{comparison of 0 <= unsigned enum expression is always true}}
    return 0;
  if (b <= 0)
    return 0;
  if (0 > b) // expected-warning {{comparison of 0 > unsigned enum expression is always false}}
    return 0;
  if (b >= 0) // expected-warning {{comparison of unsigned enum expression >= 0 is always true}}
    return 0;
  if (0 < b)
    return 0;

  if (b < 0U) // expected-warning {{comparison of unsigned enum expression < 0 is always false}}
    return 0;
  if (0U >= b)
    return 0;
  if (b > 0U)
    return 0;
  if (0U <= b) // expected-warning {{comparison of 0 <= unsigned enum expression is always true}}
    return 0;
  if (b <= 0U)
    return 0;
  if (0U > b) // expected-warning {{comparison of 0 > unsigned enum expression is always false}}
    return 0;
  if (b >= 0U) // expected-warning {{comparison of unsigned enum expression >= 0 is always true}}
    return 0;
  if (0U < b)
    return 0;

  if (c < 0)
    return 0;
  if (0 >= c) // expected-warning {{comparison 0 >= 'enum C' is always true}}
    return 0;
  if (c > 0) // expected-warning {{comparison 'enum C' > 0 is always false}}
    return 0;
  if (0 <= c)
    return 0;
  if (c <= 0) // expected-warning {{comparison 'enum C' <= 0 is always true}}
    return 0;
  if (0 > c)
    return 0;
  if (c >= 0)
    return 0;
  if (0 < c) // expected-warning {{0 < 'enum C' is always false}}
    return 0;

  if (c < 0U) // expected-warning {{comparison of unsigned enum expression < 0 is always false}}
    return 0;
  if (0U >= c)
    return 0;
  if (c > 0U)
    return 0;
  if (0U <= c) // expected-warning {{comparison of 0 <= unsigned enum expression is always true}}
    return 0;
  if (c <= 0U)
    return 0;
  if (0U > c) // expected-warning {{comparison of 0 > unsigned enum expression is always false}}
    return 0;
  if (c >= 0U) // expected-warning {{comparison of unsigned enum expression >= 0 is always true}}
    return 0;
  if (0U < c)
    return 0;
#else
  if (a < 0)
    return 0;
  if (0 >= a) // expected-warning {{comparison 0 >= 'enum A' is always true}}
    return 0;
  if (a > 0) // expected-warning {{comparison 'enum A' > 0 is always false}}
    return 0;
  if (0 <= a)
    return 0;
  if (a <= 0) // expected-warning {{comparison 'enum A' <= 0 is always true}}
    return 0;
  if (0 > a)
    return 0;
  if (a >= 0)
    return 0;
  if (0 < a) // expected-warning {{comparison 0 < 'enum A' is always false}}
    return 0;

  if (a < 0U)
    return 0;
  if (0U >= a)
    return 0;
  if (a > 0U)
    return 0;
  if (0U <= a)
    return 0;
  if (a <= 0U)
    return 0;
  if (0U > a)
    return 0;
  if (a >= 0U)
    return 0;
  if (0U < a)
    return 0;

  if (b < 0)
    return 0;
  if (0 >= b)
    return 0;
  if (b > 0)
    return 0;
  if (0 <= b)
    return 0;
  if (b <= 0)
    return 0;
  if (0 > b)
    return 0;
  if (b >= 0)
    return 0;
  if (0 < b)
    return 0;

  if (b < 0U)
    return 0;
  if (0U >= b)
    return 0;
  if (b > 0U)
    return 0;
  if (0U <= b)
    return 0;
  if (b <= 0U)
    return 0;
  if (0U > b)
    return 0;
  if (b >= 0U)
    return 0;
  if (0U < b)
    return 0;

  if (c < 0)
    return 0;
  if (0 >= c) // expected-warning {{comparison 0 >= 'enum C' is always true}}
    return 0;
  if (c > 0) // expected-warning {{comparison 'enum C' > 0 is always false}}
    return 0;
  if (0 <= c)
    return 0;
  if (c <= 0) // expected-warning {{comparison 'enum C' <= 0 is always true}}
    return 0;
  if (0 > c)
    return 0;
  if (c >= 0)
    return 0;
  if (0 < c) // expected-warning {{0 < 'enum C' is always false}}
    return 0;

  if (c < 0U)
    return 0;
  if (0U >= c)
    return 0;
  if (c > 0U)
    return 0;
  if (0U <= c)
    return 0;
  if (c <= 0U)
    return 0;
  if (0U > c)
    return 0;
  if (c >= 0U)
    return 0;
  if (0U < c)
    return 0;
#endif

  return 1;
}

namespace crash_enum_zero_width {
int test() {
  enum A : unsigned {
    A_foo = 0
  };
  enum A a;

  // used to crash in llvm::APSInt::getMaxValue()
#ifndef SILENCE
  if (a < 0) // expected-warning {{comparison of unsigned enum expression < 0 is always false}}
#else
  if (a > 0)
#endif
    return 0;

  return 1;
}
} // namespace crash_enum_zero_width
