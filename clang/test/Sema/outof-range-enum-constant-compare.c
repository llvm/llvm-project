// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -fsyntax-only -DUNSIGNED -verify %s
// RUN: %clang_cc1 -triple=x86_64-pc-win32 -fsyntax-only -DSIGNED -verify %s
// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -fsyntax-only -DUNSIGNED -DSILENCE -Wno-tautological-constant-out-of-range-compare -verify %s
// RUN: %clang_cc1 -triple=x86_64-pc-win32 -fsyntax-only -DSIGNED -DSILENCE -Wno-tautological-constant-out-of-range-compare -verify %s

int main(void) {
  enum A { A_a = 2 };
  enum A a;

#ifdef SILENCE
  // expected-no-diagnostics
#endif

#ifdef UNSIGNED
#ifndef SILENCE
  if (a < 4294967296) // expected-warning {{comparison of constant 4294967296 with expression of type 'enum A' is always true}}
    return 0;
  if (4294967296 >= a) // expected-warning {{comparison of constant 4294967296 with expression of type 'enum A' is always true}}
    return 0;
  if (a > 4294967296) // expected-warning {{comparison of constant 4294967296 with expression of type 'enum A' is always false}}
    return 0;
  if (4294967296 <= a) // expected-warning {{comparison of constant 4294967296 with expression of type 'enum A' is always false}}
    return 0;
  if (a <= 4294967296) // expected-warning {{comparison of constant 4294967296 with expression of type 'enum A' is always true}}
    return 0;
  if (4294967296 > a) // expected-warning {{comparison of constant 4294967296 with expression of type 'enum A' is always true}}
    return 0;
  if (a >= 4294967296) // expected-warning {{comparison of constant 4294967296 with expression of type 'enum A' is always false}}
    return 0;
  if (4294967296 < a) // expected-warning {{comparison of constant 4294967296 with expression of type 'enum A' is always false}}
    return 0;
  if (a == 4294967296) // expected-warning {{comparison of constant 4294967296 with expression of type 'enum A' is always false}}
    return 0;
  if (4294967296 != a) // expected-warning {{comparison of constant 4294967296 with expression of type 'enum A' is always true}}
    return 0;
  if (a != 4294967296) // expected-warning {{comparison of constant 4294967296 with expression of type 'enum A' is always true}}
    return 0;
  if (4294967296 == a) // expected-warning {{comparison of constant 4294967296 with expression of type 'enum A' is always false}}
    return 0;

  if (a < 4294967296U) // expected-warning {{comparison of constant 4294967296 with expression of type 'enum A' is always true}}
    return 0;
  if (4294967296U >= a) // expected-warning {{comparison of constant 4294967296 with expression of type 'enum A' is always true}}
    return 0;
  if (a > 4294967296U) // expected-warning {{comparison of constant 4294967296 with expression of type 'enum A' is always false}}
    return 0;
  if (4294967296U <= a) // expected-warning {{comparison of constant 4294967296 with expression of type 'enum A' is always false}}
    return 0;
  if (a <= 4294967296U) // expected-warning {{comparison of constant 4294967296 with expression of type 'enum A' is always true}}
    return 0;
  if (4294967296U > a) // expected-warning {{comparison of constant 4294967296 with expression of type 'enum A' is always true}}
    return 0;
  if (a >= 4294967296U) // expected-warning {{comparison of constant 4294967296 with expression of type 'enum A' is always false}}
    return 0;
  if (4294967296U < a) // expected-warning {{comparison of constant 4294967296 with expression of type 'enum A' is always false}}
    return 0;
  if (a == 4294967296U) // expected-warning {{comparison of constant 4294967296 with expression of type 'enum A' is always false}}
    return 0;
  if (4294967296U != a) // expected-warning {{comparison of constant 4294967296 with expression of type 'enum A' is always true}}
    return 0;
  if (a != 4294967296U) // expected-warning {{comparison of constant 4294967296 with expression of type 'enum A' is always true}}
    return 0;
  if (4294967296U == a) // expected-warning {{comparison of constant 4294967296 with expression of type 'enum A' is always false}}
    return 0;
#else // SILENCE
  if (a < 4294967296)
    return 0;
  if (4294967296 >= a)
    return 0;
  if (a > 4294967296)
    return 0;
  if (4294967296 <= a)
    return 0;
  if (a <= 4294967296)
    return 0;
  if (4294967296 > a)
    return 0;
  if (a >= 4294967296)
    return 0;
  if (4294967296 < a)
    return 0;
  if (a == 4294967296)
    return 0;
  if (4294967296 != a)
    return 0;
  if (a != 4294967296)
    return 0;
  if (4294967296 == a)
    return 0;

  if (a < 4294967296U)
    return 0;
  if (4294967296U >= a)
    return 0;
  if (a > 4294967296U)
    return 0;
  if (4294967296U <= a)
    return 0;
  if (a <= 4294967296U)
    return 0;
  if (4294967296U > a)
    return 0;
  if (a >= 4294967296U)
    return 0;
  if (4294967296U < a)
    return 0;
  if (a == 4294967296U)
    return 0;
  if (4294967296U != a)
    return 0;
  if (a != 4294967296U)
    return 0;
  if (4294967296U == a)
    return 0;
#endif
#elif defined(SIGNED)
#ifndef SILENCE
  if (a < -2147483649) // expected-warning {{comparison of constant -2147483649 with expression of type 'enum A' is always false}}
    return 0;
  if (-2147483649 >= a) // expected-warning {{comparison of constant -2147483649 with expression of type 'enum A' is always false}}
    return 0;
  if (a > -2147483649) // expected-warning {{comparison of constant -2147483649 with expression of type 'enum A' is always true}}
    return 0;
  if (-2147483649 <= a) // expected-warning {{comparison of constant -2147483649 with expression of type 'enum A' is always true}}
    return 0;
  if (a <= -2147483649) // expected-warning {{comparison of constant -2147483649 with expression of type 'enum A' is always false}}
    return 0;
  if (-2147483649 > a) // expected-warning {{comparison of constant -2147483649 with expression of type 'enum A' is always false}}
    return 0;
  if (a >= -2147483649) // expected-warning {{comparison of constant -2147483649 with expression of type 'enum A' is always true}}
    return 0;
  if (-2147483649 < a) // expected-warning {{comparison of constant -2147483649 with expression of type 'enum A' is always true}}
    return 0;
  if (a == -2147483649) // expected-warning {{comparison of constant -2147483649 with expression of type 'enum A' is always false}}
    return 0;
  if (-2147483649 != a) // expected-warning {{comparison of constant -2147483649 with expression of type 'enum A' is always true}}
    return 0;
  if (a != -2147483649) // expected-warning {{comparison of constant -2147483649 with expression of type 'enum A' is always true}}
    return 0;
  if (-2147483649 == a) // expected-warning {{comparison of constant -2147483649 with expression of type 'enum A' is always false}}
    return 0;

  if (a < 2147483648) // expected-warning {{comparison of constant 2147483648 with expression of type 'enum A' is always true}}
    return 0;
  if (2147483648 >= a) // expected-warning {{comparison of constant 2147483648 with expression of type 'enum A' is always true}}
    return 0;
  if (a > 2147483648) // expected-warning {{comparison of constant 2147483648 with expression of type 'enum A' is always false}}
    return 0;
  if (2147483648 <= a) // expected-warning {{comparison of constant 2147483648 with expression of type 'enum A' is always false}}
    return 0;
  if (a <= 2147483648) // expected-warning {{comparison of constant 2147483648 with expression of type 'enum A' is always true}}
    return 0;
  if (2147483648 > a) // expected-warning {{comparison of constant 2147483648 with expression of type 'enum A' is always true}}
    return 0;
  if (a >= 2147483648) // expected-warning {{comparison of constant 2147483648 with expression of type 'enum A' is always false}}
    return 0;
  if (2147483648 < a) // expected-warning {{comparison of constant 2147483648 with expression of type 'enum A' is always false}}
    return 0;
  if (a == 2147483648) // expected-warning {{comparison of constant 2147483648 with expression of type 'enum A' is always false}}
    return 0;
  if (2147483648 != a) // expected-warning {{comparison of constant 2147483648 with expression of type 'enum A' is always true}}
    return 0;
  if (a != 2147483648) // expected-warning {{comparison of constant 2147483648 with expression of type 'enum A' is always true}}
    return 0;
  if (2147483648 == a) // expected-warning {{comparison of constant 2147483648 with expression of type 'enum A' is always false}}
    return 0;
#else // SILENCE
  if (a < -2147483649)
    return 0;
  if (-2147483649 >= a)
    return 0;
  if (a > -2147483649)
    return 0;
  if (-2147483649 <= a)
    return 0;
  if (a <= -2147483649)
    return 0;
  if (-2147483649 > a)
    return 0;
  if (a >= -2147483649)
    return 0;
  if (-2147483649 < a)
    return 0;
  if (a == -2147483649)
    return 0;
  if (-2147483649 != a)
    return 0;
  if (a != -2147483649)
    return 0;
  if (-2147483649 == a)
    return 0;

  if (a < 2147483648)
    return 0;
  if (2147483648 >= a)
    return 0;
  if (a > 2147483648)
    return 0;
  if (2147483648 <= a)
    return 0;
  if (a <= 2147483648)
    return 0;
  if (2147483648 > a)
    return 0;
  if (a >= 2147483648)
    return 0;
  if (2147483648 < a)
    return 0;
  if (a == 2147483648)
    return 0;
  if (2147483648 != a)
    return 0;
  if (a != 2147483648)
    return 0;
  if (2147483648 == a)
    return 0;
#endif
#endif
}

// https://bugs.llvm.org/show_bug.cgi?id=35009
int PR35009(void) {
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
