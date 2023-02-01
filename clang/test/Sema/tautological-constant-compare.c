// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -Wtautological-constant-in-range-compare -DTEST=2 -verify %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -Wtautological-constant-in-range-compare -DTEST=2 -verify -x c++ %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -Wtautological-type-limit-compare -DTEST -verify %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -Wtautological-type-limit-compare -DTEST -verify -x c++ %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -Wtype-limits -DTEST -verify %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -Wtype-limits -DTEST -verify -x c++ %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -Wextra -Wno-sign-compare -DTEST -verify %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -Wextra -Wno-sign-compare -DTEST -verify -x c++ %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -Wall -verify=silent %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -Wall -verify=silent -x c++ %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -verify=silent %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -verify=silent -x c++ %s

#ifndef TEST
// silent-no-diagnostics
#endif

int value(void);

#define macro(val) val

#ifdef __cplusplus
template<typename T>
void TFunc() {
  // Make sure that we do warn for normal variables in template functions !
  unsigned char c = value();
#ifdef TEST
  if (c > 255) // expected-warning {{comparison 'unsigned char' > 255 is always false}}
      return;
#else
  if (c > 255)
      return;
#endif

  if (c > macro(255))
      return;

  T v = value();
  if (v > 255)
      return;
  if (v > 32767)
      return;
}
#endif

int main(void)
{
#ifdef __cplusplus
  TFunc<unsigned char>();
  TFunc<signed short>();
#endif

  short s = value();

#ifdef TEST
  if (s == 32767)
      return 0;
  if (s != 32767)
      return 0;
  if (s < 32767)
      return 0;
  if (s <= 32767) // expected-warning {{comparison 'short' <= 32767 is always true}}
      return 0;
  if (s > 32767) // expected-warning {{comparison 'short' > 32767 is always false}}
      return 0;
  if (s >= 32767)
      return 0;

  if (32767 == s)
      return 0;
  if (32767 != s)
      return 0;
  if (32767 < s) // expected-warning {{comparison 32767 < 'short' is always false}}
      return 0;
  if (32767 <= s)
      return 0;
  if (32767 > s)
      return 0;
  if (32767 >= s) // expected-warning {{comparison 32767 >= 'short' is always true}}
      return 0;

  // FIXME: assumes two's complement
  if (s == -32768)
      return 0;
  if (s != -32768)
      return 0;
  if (s < -32768) // expected-warning {{comparison 'short' < -32768 is always false}}
      return 0;
  if (s <= -32768)
      return 0;
  if (s > -32768)
      return 0;
  if (s >= -32768) // expected-warning {{comparison 'short' >= -32768 is always true}}
      return 0;

  if (-32768 == s)
      return 0;
  if (-32768 != s)
      return 0;
  if (-32768 < s)
      return 0;
  if (-32768 <= s) // expected-warning {{comparison -32768 <= 'short' is always true}}
      return 0;
  if (-32768 > s) // expected-warning {{comparison -32768 > 'short' is always false}}
      return 0;
  if (-32768 >= s)
      return 0;

  // Note: both sides are promoted to unsigned long prior to the comparison, so
  // it is perfectly possible for a short to compare greater than 32767UL.
  if (s == 32767UL)
      return 0;
  if (s != 32767UL)
      return 0;
  if (s < 32767UL)
      return 0;
  if (s <= 32767UL)
      return 0;
  if (s > 32767UL)
      return 0;
  if (s >= 32767UL)
      return 0;

  if (32767UL == s)
      return 0;
  if (32767UL != s)
      return 0;
  if (32767UL < s)
      return 0;
  if (32767UL <= s)
      return 0;
  if (32767UL > s)
      return 0;
  if (32767UL >= s)
      return 0;

  enum { ULONG_MAX = (2UL * (unsigned long)__LONG_MAX__ + 1UL) };
  if (s == 2UL * (unsigned long)__LONG_MAX__ + 1UL)
      return 0;
  if (s != 2UL * (unsigned long)__LONG_MAX__ + 1UL)
      return 0;
  if (s < 2UL * (unsigned long)__LONG_MAX__ + 1UL)
      return 0;
  if (s <= 2UL * (unsigned long)__LONG_MAX__ + 1UL) // expected-warning-re {{comparison 'short' <= {{.*}} is always true}}
      return 0;
  if (s > 2UL * (unsigned long)__LONG_MAX__ + 1UL) // expected-warning-re {{comparison 'short' > {{.*}} is always false}}
      return 0;
  if (s >= 2UL * (unsigned long)__LONG_MAX__ + 1UL)
      return 0;

  if (2UL * (unsigned long)__LONG_MAX__ + 1UL == s)
      return 0;
  if (2UL * (unsigned long)__LONG_MAX__ + 1UL != s)
      return 0;
  if (2UL * (unsigned long)__LONG_MAX__ + 1UL < s) // expected-warning-re {{comparison {{.*}} < 'short' is always false}}
      return 0;
  if (2UL * (unsigned long)__LONG_MAX__ + 1UL <= s)
      return 0;
  if (2UL * (unsigned long)__LONG_MAX__ + 1UL > s)
      return 0;
  if (2UL * (unsigned long)__LONG_MAX__ + 1UL >= s) // expected-warning-re {{comparison {{.*}} >= 'short' is always true}}
      return 0;

  // FIXME: assumes two's complement
  if (s == -32768L)
      return 0;
  if (s != -32768L)
      return 0;
  if (s < -32768L) // expected-warning {{comparison 'short' < -32768 is always false}}
      return 0;
  if (s <= -32768L)
      return 0;
  if (s > -32768L)
      return 0;
  if (s >= -32768L) // expected-warning {{comparison 'short' >= -32768 is always true}}
      return 0;

  if (-32768L == s)
      return 0;
  if (-32768L != s)
      return 0;
  if (-32768L < s)
      return 0;
  if (-32768L <= s) // expected-warning {{comparison -32768 <= 'short' is always true}}
      return 0;
  if (-32768L > s) // expected-warning {{comparison -32768 > 'short' is always false}}
      return 0;
  if (-32768L >= s)
      return 0;
#else
  if (s == 32767)
    return 0;
  if (s != 32767)
    return 0;
  if (s < 32767)
    return 0;
  if (s <= 32767)
    return 0;
  if (s > 32767)
    return 0;
  if (s >= 32767)
    return 0;

  if (32767 == s)
    return 0;
  if (32767 != s)
    return 0;
  if (32767 < s)
    return 0;
  if (32767 <= s)
    return 0;
  if (32767 > s)
    return 0;
  if (32767 >= s)
    return 0;

  // FIXME: assumes two's complement
  if (s == -32768)
    return 0;
  if (s != -32768)
    return 0;
  if (s < -32768)
    return 0;
  if (s <= -32768)
    return 0;
  if (s > -32768)
    return 0;
  if (s >= -32768)
    return 0;

  if (-32768 == s)
    return 0;
  if (-32768 != s)
    return 0;
  if (-32768 < s)
    return 0;
  if (-32768 <= s)
    return 0;
  if (-32768 > s)
    return 0;
  if (-32768 >= s)
    return 0;

  if (s == 32767UL)
    return 0;
  if (s != 32767UL)
    return 0;
  if (s < 32767UL)
    return 0;
  if (s <= 32767UL)
    return 0;
  if (s > 32767UL)
    return 0;
  if (s >= 32767UL)
    return 0;

  if (32767UL == s)
    return 0;
  if (32767UL != s)
    return 0;
  if (32767UL < s)
    return 0;
  if (32767UL <= s)
    return 0;
  if (32767UL > s)
    return 0;
  if (32767UL >= s)
    return 0;

  // FIXME: assumes two's complement
  if (s == -32768L)
    return 0;
  if (s != -32768L)
    return 0;
  if (s < -32768L)
    return 0;
  if (s <= -32768L)
    return 0;
  if (s > -32768L)
    return 0;
  if (s >= -32768L)
    return 0;

  if (-32768L == s)
    return 0;
  if (-32768L != s)
    return 0;
  if (-32768L < s)
    return 0;
  if (-32768L <= s)
    return 0;
  if (-32768L > s)
    return 0;
  if (-32768L >= s)
    return 0;
#endif

  if (s == 0)
    return 0;
  if (s != 0)
    return 0;
  if (s < 0)
    return 0;
  if (s <= 0)
    return 0;
  if (s > 0)
    return 0;
  if (s >= 0)
    return 0;

  if (0 == s)
    return 0;
  if (0 != s)
    return 0;
  if (0 < s)
    return 0;
  if (0 <= s)
    return 0;
  if (0 > s)
    return 0;
  if (0 >= s)
    return 0;

  unsigned short us = value();

#ifdef TEST
  if (us == 65535)
      return 0;
  if (us != 65535)
      return 0;
  if (us < 65535)
      return 0;
  if (us <= 65535) // expected-warning {{comparison 'unsigned short' <= 65535 is always true}}
      return 0;
  if (us > 65535) // expected-warning {{comparison 'unsigned short' > 65535 is always false}}
      return 0;
  if (us >= 65535)
      return 0;

  if (65535 == us)
      return 0;
  if (65535 != us)
      return 0;
  if (65535 < us) // expected-warning {{comparison 65535 < 'unsigned short' is always false}}
      return 0;
  if (65535 <= us)
      return 0;
  if (65535 > us)
      return 0;
  if (65535 >= us) // expected-warning {{comparison 65535 >= 'unsigned short' is always true}}
      return 0;

  if (us == 65535UL)
      return 0;
  if (us != 65535UL)
      return 0;
  if (us < 65535UL)
      return 0;
  if (us <= 65535UL) // expected-warning {{comparison 'unsigned short' <= 65535 is always true}}
      return 0;
  if (us > 65535UL) // expected-warning {{comparison 'unsigned short' > 65535 is always false}}
      return 0;
  if (us >= 65535UL)
      return 0;

  if (65535UL == us)
      return 0;
  if (65535UL != us)
      return 0;
  if (65535UL < us) // expected-warning {{comparison 65535 < 'unsigned short' is always false}}
      return 0;
  if (65535UL <= us)
      return 0;
  if (65535UL > us)
      return 0;
  if (65535UL >= us) // expected-warning {{comparison 65535 >= 'unsigned short' is always true}}
      return 0;
#else
  if (us == 65535)
      return 0;
  if (us != 65535)
      return 0;
  if (us < 65535)
      return 0;
  if (us <= 65535)
      return 0;
  if (us > 65535)
      return 0;
  if (us >= 65535)
      return 0;

  if (65535 == us)
      return 0;
  if (65535 != us)
      return 0;
  if (65535 < us)
      return 0;
  if (65535 <= us)
      return 0;
  if (65535 > us)
      return 0;
  if (65535 >= us)
      return 0;

  if (us == 65535UL)
      return 0;
  if (us != 65535UL)
      return 0;
  if (us < 65535UL)
      return 0;
  if (us <= 65535UL)
      return 0;
  if (us > 65535UL)
      return 0;
  if (us >= 65535UL)
      return 0;

  if (65535UL == us)
      return 0;
  if (65535UL != us)
      return 0;
  if (65535UL < us)
      return 0;
  if (65535UL <= us)
      return 0;
  if (65535UL > us)
      return 0;
  if (65535UL >= us)
      return 0;
#endif

  if (us == 32767)
    return 0;
  if (us != 32767)
    return 0;
  if (us < 32767)
    return 0;
  if (us <= 32767)
    return 0;
  if (us > 32767)
    return 0;
  if (us >= 32767)
    return 0;

  if (32767 == us)
    return 0;
  if (32767 != us)
    return 0;
  if (32767 < us)
    return 0;
  if (32767 <= us)
    return 0;
  if (32767 > us)
    return 0;
  if (32767 >= us)
    return 0;

  if (us == 32767UL)
    return 0;
  if (us != 32767UL)
    return 0;
  if (us < 32767UL)
    return 0;
  if (us <= 32767UL)
    return 0;
  if (us > 32767UL)
    return 0;
  if (us >= 32767UL)
    return 0;

  if (32767UL == us)
    return 0;
  if (32767UL != us)
    return 0;
  if (32767UL < us)
    return 0;
  if (32767UL <= us)
    return 0;
  if (32767UL > us)
    return 0;
  if (32767UL >= us)
    return 0;

#if __SIZEOF_INT128__
  __int128 i128 = value();
  if (i128 == -1) // used to crash
      return 0;
#endif


  enum E {
  yes,
  no,
  maybe
  };
  enum E e = (enum E)value();

  if (e == yes)
      return 0;
  if (e != yes)
      return 0;
  if (e < yes)
      return 0;
  if (e <= yes)
      return 0;
  if (e > yes)
      return 0;
  if (e >= yes)
      return 0;

  if (yes == e)
      return 0;
  if (yes != e)
      return 0;
  if (yes < e)
      return 0;
  if (yes <= e)
      return 0;
  if (yes > e)
      return 0;
  if (yes >= e)
      return 0;

  if (e == maybe)
      return 0;
  if (e != maybe)
      return 0;
  if (e < maybe)
      return 0;
  if (e <= maybe)
      return 0;
  if (e > maybe)
      return 0;
  if (e >= maybe)
      return 0;

  if (maybe == e)
      return 0;
  if (maybe != e)
      return 0;
  if (maybe < e)
      return 0;
  if (maybe <= e)
      return 0;
  if (maybe > e)
      return 0;
  if (maybe >= e)
      return 0;

  // We only warn on out-of-range bitfields and expressions with limited range
  // under -Wtantological-in-range-compare, not under -Wtype-limits, because
  // the warning is not based on the type alone.
  struct A {
    int a : 3;
    unsigned b : 3;
    long c : 3;
    unsigned long d : 3;
  } a;
  if (a.a < 3) {}
  if (a.a < 4) {} // #bitfield1
  if (a.b < 7) {}
  if (a.b < 8) {} // #bitfield2
  if (a.c < 3) {}
  if (a.c < 4) {} // #bitfield3
  if (a.d < 7) {}
  if (a.d < 8) {} // #bitfield4
#if TEST == 2
  // expected-warning@#bitfield1 {{comparison of 3-bit signed value < 4 is always true}}
  // expected-warning@#bitfield2 {{comparison of 3-bit unsigned value < 8 is always true}}
  // expected-warning@#bitfield3 {{comparison of 3-bit signed value < 4 is always true}}
  // expected-warning@#bitfield4 {{comparison of 3-bit unsigned value < 8 is always true}}
#endif

  if ((s & 0xff) < 0) {} // #valuerange1
  if ((s & 0xff) < 1) {}
  if ((s & -3) < -4) {}
  if ((s & -3) < -3) {}
  if ((s & -3) < 4u) {}
  if ((s & -3) > 4u) {}
  if ((s & -3) == 4u) {}
  if ((s & -3) == 3u) {} // FIXME: Impossible.
  if ((s & -3) == -5u) {}
  if ((s & -3) == -4u) {}
#if TEST == 2
  // expected-warning@#valuerange1 {{comparison of 8-bit unsigned value < 0 is always false}}
#endif

  // FIXME: Our bit-level width tracking comes unstuck here: the second of the
  // conditions below is also tautological, but we can't tell that because we
  // don't track the actual range, only the bit-width.
  if ((s ? 1 : 0) + (us ? 1 : 0) > 1) {}
  if ((s ? 1 : 0) + (us ? 1 : 0) > 2) {}
  if ((s ? 1 : 0) + (us ? 1 : 0) > 3) {} // #addrange1
#if TEST == 2
  // expected-warning@#addrange1 {{comparison of 2-bit unsigned value > 3 is always false}}
#endif

  // FIXME: The second and third comparisons are also tautological; 0x40000000
  // is the greatest value that multiplying two int16s can produce.
  if (s * s > 0x3fffffff) {}
  if (s * s > 0x40000000) {}
  if (s * s > 0x7ffffffe) {}
  if (s * s > 0x7fffffff) {} // expected-warning {{result of comparison 'int' > 2147483647 is always false}}

  if ((s & 0x3ff) * (s & 0x1f) > 0x7be0) {}
  if ((s & 0x3ff) * (s & 0x1f) > 0x7be1) {} // FIXME
  if ((s & 0x3ff) * (s & 0x1f) > 0x7ffe) {} // FIXME
  if ((s & 0x3ff) * (s & 0x1f) > 0x7fff) {} // #mulrange1
#if TEST == 2
  // expected-warning@#mulrange1 {{comparison of 15-bit unsigned value > 32767 is always false}}
#endif

  if (a.a * a.b > 21) {} // FIXME
  if (a.a * a.b > 31) {} // #mulrange2
#if TEST == 2
  // expected-warning@#mulrange2 {{comparison of 6-bit signed value > 31 is always false}}
#endif

  if (a.a - (s & 1) < -4) {}
  if (a.a - (s & 1) < -7) {} // FIXME
  if (a.a - (s & 1) < -8) {} // #subrange1
  if (a.a - (s & 1) > 3) {} // FIXME: Can be < -4 but not > 3.
  if (a.a - (s & 1) > 7) {} // #subrange2

  if (a.a - (s & 7) < -8) {}
  if (a.a - (s & 7) > 7) {} // FIXME: Can be < -8 but not > 7.
  if (a.a - (s & 7) < -15) {}
  if (a.a - (s & 7) < -16) {} // #subrange3
  if (a.a - (s & 7) > 15) {} // #subrange4

  if (a.b - (s & 1) > 6) {}
  if (a.b - (s & 1) > 7) {} // #subrange5
  if (a.b - (s & 7) < -8) {} // #subrange6
  if (a.b - (s & 15) < -8) {}
  if (a.b - (s & 15) < -16) {} // #subrange7
#if TEST == 2
  // expected-warning@#subrange1 {{comparison of 4-bit signed value < -8 is always false}}
  // expected-warning@#subrange2 {{comparison of 4-bit signed value > 7 is always false}}
  // expected-warning@#subrange3 {{comparison of 5-bit signed value < -16 is always false}}
  // expected-warning@#subrange4 {{comparison of 5-bit signed value > 15 is always false}}
  // expected-warning@#subrange5 {{comparison of 4-bit signed value > 7 is always false}}
  // expected-warning@#subrange6 {{comparison of 4-bit signed value < -8 is always false}}
  // expected-warning@#subrange7 {{comparison of 5-bit signed value < -16 is always false}}
#endif

  // a.a % 3 is in range [-2, 2], which we expand to [-4, 4)
  if (a.a % 3 > 2) {}
  if (a.a % 3 > 3) {} // #remrange1
  if (a.a % 3 == -1) {}
  if (a.a % 3 == -2) {}
  if (a.a % 3 < -3) {} // FIXME
  if (a.a % 3 < -4) {} // #remrange2

  // a.b % 3 is in range [0, 3), which we expand to [0, 4)
  if (a.b % 3 > 2) {}
  if (a.b % 3 > 3) {} // #remrange3
  if (a.b % 3 < 0) {} // #remrange4
#if TEST == 2
  // expected-warning@#remrange1 {{comparison of 3-bit signed value > 3 is always false}}
  // expected-warning@#remrange2 {{comparison of 3-bit signed value < -4 is always false}}
  // expected-warning@#remrange3 {{comparison of 2-bit unsigned value > 3 is always false}}
  // expected-warning@#remrange4 {{comparison of 2-bit unsigned value < 0 is always false}}
#endif

  // Don't warn on non-constant-expression values that end up being a constant
  // 0; we generally only want to warn when one side of the comparison is
  // effectively non-constant.
  if ("x"[1] == 0) {}
  if (((void)s, 0) == 0) {}

  return 1;
}
