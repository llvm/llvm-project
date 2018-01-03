// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -fsyntax-only -DUNSIGNED -Wtautological-constant-in-range-compare -verify %s
// RUN: %clang_cc1 -triple=x86_64-pc-win32 -fsyntax-only -DSIGNED -Wtautological-constant-in-range-compare -verify %s
// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -fsyntax-only -DUNSIGNED -DSILENCE -Wno-tautological-constant-compare -verify %s
// RUN: %clang_cc1 -triple=x86_64-pc-win32 -fsyntax-only -DSIGNED -DSILENCE -Wno-tautological-constant-compare -verify %s

int main() {
  enum A { A_a = 2 };
  enum A a;

#ifdef SILENCE
  // expected-no-diagnostics
#else
  // If we promote to unsigned, it doesn't matter whether the enum's underlying
  // type was signed.
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

  if (a < 4294967295U)
    return 0;
  if (4294967295U >= a) // expected-warning {{comparison 4294967295 >= 'enum A' is always true}}
    return 0;
  if (a > 4294967295U) // expected-warning {{comparison 'enum A' > 4294967295 is always false}}
    return 0;
  if (4294967295U <= a)
    return 0;
  if (a <= 4294967295U) // expected-warning {{comparison 'enum A' <= 4294967295 is always true}}
    return 0;
  if (4294967295U > a)
    return 0;
  if (a >= 4294967295U)
    return 0;
  if (4294967295U < a) // expected-warning {{comparison 4294967295 < 'enum A' is always false}}
    return 0;

  if (a < 2147483647U)
    return 0;
  if (2147483647U >= a)
    return 0;
  if (a > 2147483647U)
    return 0;
  if (2147483647U <= a)
    return 0;
  if (a <= 2147483647U)
    return 0;
  if (2147483647U > a)
    return 0;
  if (a >= 2147483647U)
    return 0;
  if (2147483647U < a)
    return 0;
#endif

#if defined(UNSIGNED) && !defined(SILENCE)
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

  if (a < 4294967295)
    return 0;
  if (4294967295 >= a) // expected-warning {{comparison 4294967295 >= 'enum A' is always true}}
    return 0;
  if (a > 4294967295) // expected-warning {{comparison 'enum A' > 4294967295 is always false}}
    return 0;
  if (4294967295 <= a)
    return 0;
  if (a <= 4294967295) // expected-warning {{comparison 'enum A' <= 4294967295 is always true}}
    return 0;
  if (4294967295 > a)
    return 0;
  if (a >= 4294967295)
    return 0;
  if (4294967295 < a) // expected-warning {{comparison 4294967295 < 'enum A' is always false}}
    return 0;
#else // SIGNED || SILENCE
  if (a < 0)
    return 0;
  if (0 >= a)
    return 0;
  if (a > 0)
    return 0;
  if (0 <= a)
    return 0;
  if (a <= 0)
    return 0;
  if (0 > a)
    return 0;
  if (a >= 0)
    return 0;
  if (0 < a)
    return 0;

#ifndef SILENCE
  if (a < 4294967295) // expected-warning {{comparison of constant 4294967295 with expression of type 'enum A' is always true}}
    return 0;
  if (4294967295 >= a) // expected-warning {{comparison of constant 4294967295 with expression of type 'enum A' is always true}}
    return 0;
  if (a > 4294967295) // expected-warning {{comparison of constant 4294967295 with expression of type 'enum A' is always false}}
    return 0;
  if (4294967295 <= a) // expected-warning {{comparison of constant 4294967295 with expression of type 'enum A' is always false}}
    return 0;
  if (a <= 4294967295) // expected-warning {{comparison of constant 4294967295 with expression of type 'enum A' is always true}}
    return 0;
  if (4294967295 > a) // expected-warning {{comparison of constant 4294967295 with expression of type 'enum A' is always true}}
    return 0;
  if (a >= 4294967295) // expected-warning {{comparison of constant 4294967295 with expression of type 'enum A' is always false}}
    return 0;
  if (4294967295 < a) // expected-warning {{comparison of constant 4294967295 with expression of type 'enum A' is always false}}
    return 0;
#else
  if (a < 4294967295)
    return 0;
  if (4294967295 >= a)
    return 0;
  if (a > 4294967295)
    return 0;
  if (4294967295 <= a)
    return 0;
  if (a <= 4294967295)
    return 0;
  if (4294967295 > a)
    return 0;
  if (a >= 4294967295)
    return 0;
  if (4294967295 < a)
    return 0;
#endif
#endif

#if defined(SIGNED) && !defined(SILENCE)
  if (a < -2147483648) // expected-warning {{comparison 'enum A' < -2147483648 is always false}}
    return 0;
  if (-2147483648 >= a)
    return 0;
  if (a > -2147483648)
    return 0;
  if (-2147483648 <= a) // expected-warning {{comparison -2147483648 <= 'enum A' is always true}}
    return 0;
  if (a <= -2147483648)
    return 0;
  if (-2147483648 > a) // expected-warning {{comparison -2147483648 > 'enum A' is always false}}
    return 0;
  if (a >= -2147483648) // expected-warning {{comparison 'enum A' >= -2147483648 is always true}}
    return 0;
  if (-2147483648 < a)
    return 0;

  if (a < 2147483647)
    return 0;
  if (2147483647 >= a) // expected-warning {{comparison 2147483647 >= 'enum A' is always true}}
    return 0;
  if (a > 2147483647) // expected-warning {{comparison 'enum A' > 2147483647 is always false}}
    return 0;
  if (2147483647 <= a)
    return 0;
  if (a <= 2147483647) // expected-warning {{comparison 'enum A' <= 2147483647 is always true}}
    return 0;
  if (2147483647 > a)
    return 0;
  if (a >= 2147483647)
    return 0;
  if (2147483647 < a) // expected-warning {{comparison 2147483647 < 'enum A' is always false}}
    return 0;
#elif defined(UNSIGNED) && !defined(SILENCE)
#ifndef SILENCE
  if (a < -2147483648) // expected-warning {{comparison of constant -2147483648 with expression of type 'enum A' is always false}}
    return 0;
  if (-2147483648 >= a) // expected-warning {{comparison of constant -2147483648 with expression of type 'enum A' is always false}}
    return 0;
  if (a > -2147483648) // expected-warning {{comparison of constant -2147483648 with expression of type 'enum A' is always true}}
    return 0;
  if (-2147483648 <= a) // expected-warning {{comparison of constant -2147483648 with expression of type 'enum A' is always true}}
    return 0;
  if (a <= -2147483648) // expected-warning {{comparison of constant -2147483648 with expression of type 'enum A' is always false}}
    return 0;
  if (-2147483648 > a) // expected-warning {{comparison of constant -2147483648 with expression of type 'enum A' is always false}}
    return 0;
  if (a >= -2147483648) // expected-warning {{comparison of constant -2147483648 with expression of type 'enum A' is always true}}
    return 0;
  if (-2147483648 < a) // expected-warning {{comparison of constant -2147483648 with expression of type 'enum A' is always true}}
    return 0;
#else
  if (a < -2147483648)
    return 0;
  if (-2147483648 >= a)
    return 0;
  if (a > -2147483648)
    return 0;
  if (-2147483648 <= a)
    return 0;
  if (a <= -2147483648)
    return 0;
  if (-2147483648 > a)
    return 0;
  if (a >= -2147483648)
    return 0;
  if (-2147483648 < a)
    return 0;
#endif

  if (a < 2147483647)
    return 0;
  if (2147483647 >= a)
    return 0;
  if (a > 2147483647)
    return 0;
  if (2147483647 <= a)
    return 0;
  if (a <= 2147483647)
    return 0;
  if (2147483647 > a)
    return 0;
  if (a >= 2147483647)
    return 0;
  if (2147483647 < a)
    return 0;
#endif

  return 1;
}

// https://bugs.llvm.org/show_bug.cgi?id=35009
int PR35009() {
  enum A { A_a = 2 };
  enum A a;

  // in C, this should not warn.

  if (a < 1)
    return 0;
  if (1 >= a)
    return 0;
  if (a > 1)
    return 0;
  if (1 <= a)
    return 0;
  if (a <= 1)
    return 0;
  if (1 > a)
    return 0;
  if (a >= 1)
    return 0;
  if (1 < a)
    return 0;
  if (a == 1)
    return 0;
  if (1 != a)
    return 0;
  if (a != 1)
    return 0;
  if (1 == a)
    return 0;

  if (a < 1U)
    return 0;
  if (1U >= a)
    return 0;
  if (a > 1U)
    return 0;
  if (1U <= a)
    return 0;
  if (a <= 1U)
    return 0;
  if (1U > a)
    return 0;
  if (a >= 1U)
    return 0;
  if (1U < a)
    return 0;
  if (a == 1U)
    return 0;
  if (1U != a)
    return 0;
  if (a != 1U)
    return 0;
  if (1U == a)
    return 0;

  if (a < 2)
    return 0;
  if (2 >= a)
    return 0;
  if (a > 2)
    return 0;
  if (2 <= a)
    return 0;
  if (a <= 2)
    return 0;
  if (2 > a)
    return 0;
  if (a >= 2)
    return 0;
  if (2 < a)
    return 0;
  if (a == 2)
    return 0;
  if (2 != a)
    return 0;
  if (a != 2)
    return 0;
  if (2 == a)
    return 0;

  if (a < 2U)
    return 0;
  if (2U >= a)
    return 0;
  if (a > 2U)
    return 0;
  if (2U <= a)
    return 0;
  if (a <= 2U)
    return 0;
  if (2U > a)
    return 0;
  if (a >= 2U)
    return 0;
  if (2U < a)
    return 0;
  if (a == 2U)
    return 0;
  if (2U != a)
    return 0;
  if (a != 2U)
    return 0;
  if (2U == a)
    return 0;

  if (a < 3)
    return 0;
  if (3 >= a)
    return 0;
  if (a > 3)
    return 0;
  if (3 <= a)
    return 0;
  if (a <= 3)
    return 0;
  if (3 > a)
    return 0;
  if (a >= 3)
    return 0;
  if (3 < a)
    return 0;
  if (a == 3)
    return 0;
  if (3 != a)
    return 0;
  if (a != 3)
    return 0;
  if (3 == a)
    return 0;

  if (a < 3U)
    return 0;
  if (3U >= a)
    return 0;
  if (a > 3U)
    return 0;
  if (3U <= a)
    return 0;
  if (a <= 3U)
    return 0;
  if (3U > a)
    return 0;
  if (a >= 3U)
    return 0;
  if (3U < a)
    return 0;
  if (a == 3U)
    return 0;
  if (3U != a)
    return 0;
  if (a != 3U)
    return 0;
  if (3U == a)
    return 0;

  return 1;
}
