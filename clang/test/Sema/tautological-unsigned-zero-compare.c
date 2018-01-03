// RUN: %clang_cc1 -fsyntax-only \
// RUN:            -Wtautological-unsigned-zero-compare \
// RUN:            -verify %s
// RUN: %clang_cc1 -fsyntax-only \
// RUN:            -verify=silence %s
// RUN: %clang_cc1 -fsyntax-only \
// RUN:            -Wtautological-unsigned-zero-compare \
// RUN:            -verify -x c++ %s
// RUN: %clang_cc1 -fsyntax-only \
// RUN:            -verify=silence -x c++ %s

unsigned uvalue(void);
signed int svalue(void);

#define macro(val) val

#ifdef __cplusplus
template<typename T>
void TFunc() {
  // Make sure that we do warn for normal variables in template functions !
  unsigned char c = svalue();
  if (c < 0) // expected-warning {{comparison of unsigned expression < 0 is always false}}
      return;

  if (c < macro(0))
      return;

  T v = svalue();
  if (v < 0)
      return;
}
#endif

int main()
{
#ifdef __cplusplus
  TFunc<unsigned char>();
  TFunc<unsigned short>();
#endif

  short s = svalue();

  unsigned un = uvalue();

  // silence-no-diagnostics

  // Note: both sides are promoted to unsigned long prior to the comparison.
  if (s == 0UL)
      return 0;
  if (s != 0UL)
      return 0;
  if (s < 0UL) // expected-warning {{comparison of unsigned expression < 0 is always false}}
      return 0;
  if (s <= 0UL)
      return 0;
  if (s > 0UL)
      return 0;
  if (s >= 0UL) // expected-warning {{comparison of unsigned expression >= 0 is always true}}
      return 0;

  if (0UL == s)
      return 0;
  if (0UL != s)
      return 0;
  if (0UL < s)
      return 0;
  if (0UL <= s) // expected-warning {{comparison of 0 <= unsigned expression is always true}}
      return 0;
  if (0UL > s) // expected-warning {{comparison of 0 > unsigned expression is always false}}
      return 0;
  if (0UL >= s)
      return 0;

  if (un == 0)
      return 0;
  if (un != 0)
      return 0;
  if (un < 0) // expected-warning {{comparison of unsigned expression < 0 is always false}}
      return 0;
  if (un <= 0)
      return 0;
  if (un > 0)
      return 0;
  if (un >= 0) // expected-warning {{comparison of unsigned expression >= 0 is always true}}
      return 0;

  if (0 == un)
      return 0;
  if (0 != un)
      return 0;
  if (0 < un)
      return 0;
  if (0 <= un) // expected-warning {{comparison of 0 <= unsigned expression is always true}}
      return 0;
  if (0 > un) // expected-warning {{comparison of 0 > unsigned expression is always false}}
      return 0;
  if (0 >= un)
      return 0;

  if (un == 0UL)
      return 0;
  if (un != 0UL)
      return 0;
  if (un < 0UL) // expected-warning {{comparison of unsigned expression < 0 is always false}}
      return 0;
  if (un <= 0UL)
      return 0;
  if (un > 0UL)
      return 0;
  if (un >= 0UL) // expected-warning {{comparison of unsigned expression >= 0 is always true}}
      return 0;

  if (0UL == un)
      return 0;
  if (0UL != un)
      return 0;
  if (0UL < un)
      return 0;
  if (0UL <= un) // expected-warning {{comparison of 0 <= unsigned expression is always true}}
      return 0;
  if (0UL > un) // expected-warning {{comparison of 0 > unsigned expression is always false}}
      return 0;
  if (0UL >= un)
      return 0;


  signed int a = svalue();

  if (a == 0)
      return 0;
  if (a != 0)
      return 0;
  if (a < 0)
      return 0;
  if (a <= 0)
      return 0;
  if (a > 0)
      return 0;
  if (a >= 0)
      return 0;

  if (0 == a)
      return 0;
  if (0 != a)
      return 0;
  if (0 < a)
      return 0;
  if (0 <= a)
      return 0;
  if (0 > a)
      return 0;
  if (0 >= a)
      return 0;

  if (a == 0UL)
      return 0;
  if (a != 0UL)
      return 0;
  if (a < 0UL) // expected-warning {{comparison of unsigned expression < 0 is always false}}
      return 0;
  if (a <= 0UL)
      return 0;
  if (a > 0UL)
      return 0;
  if (a >= 0UL) // expected-warning {{comparison of unsigned expression >= 0 is always true}}
      return 0;

  if (0UL == a)
      return 0;
  if (0UL != a)
      return 0;
  if (0UL < a)
      return 0;
  if (0UL <= a) // expected-warning {{comparison of 0 <= unsigned expression is always true}}
      return 0;
  if (0UL > a) // expected-warning {{comparison of 0 > unsigned expression is always false}}
      return 0;
  if (0UL >= a)
      return 0;


  float fl = 0;

  if (fl == 0)
      return 0;
  if (fl != 0)
      return 0;
  if (fl < 0)
      return 0;
  if (fl <= 0)
      return 0;
  if (fl > 0)
      return 0;
  if (fl >= 0)
      return 0;

  if (0 == fl)
      return 0;
  if (0 != fl)
      return 0;
  if (0 < fl)
      return 0;
  if (0 <= fl)
      return 0;
  if (0 > fl)
      return 0;
  if (0 >= fl)
      return 0;

  if (fl == 0UL)
      return 0;
  if (fl != 0UL)
      return 0;
  if (fl < 0UL)
      return 0;
  if (fl <= 0UL)
      return 0;
  if (fl > 0UL)
      return 0;
  if (fl >= 0UL)
      return 0;

  if (0UL == fl)
      return 0;
  if (0UL != fl)
      return 0;
  if (0UL < fl)
      return 0;
  if (0UL <= fl)
      return 0;
  if (0UL > fl)
      return 0;
  if (0UL >= fl)
      return 0;


  double dl = 0;

  if (dl == 0)
      return 0;
  if (dl != 0)
      return 0;
  if (dl < 0)
      return 0;
  if (dl <= 0)
      return 0;
  if (dl > 0)
      return 0;
  if (dl >= 0)
      return 0;

  if (0 == dl)
      return 0;
  if (0 != dl)
      return 0;
  if (0 < dl)
      return 0;
  if (0 <= dl)
      return 0;
  if (0 > dl)
      return 0;
  if (0 >= dl)
      return 0;

  if (dl == 0UL)
      return 0;
  if (dl != 0UL)
      return 0;
  if (dl < 0UL)
      return 0;
  if (dl <= 0UL)
      return 0;
  if (dl > 0UL)
      return 0;
  if (dl >= 0UL)
      return 0;

  if (0UL == dl)
      return 0;
  if (0UL != dl)
      return 0;
  if (0UL < dl)
      return 0;
  if (0UL <= dl)
      return 0;
  if (0UL > dl)
      return 0;
  if (0UL >= dl)
      return 0;

  return 1;
}
