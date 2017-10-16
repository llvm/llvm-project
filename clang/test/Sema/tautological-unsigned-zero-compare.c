// RUN: %clang_cc1 -fsyntax-only -DTEST -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wno-tautological-unsigned-zero-compare -verify %s
// RUN: %clang_cc1 -fsyntax-only -DTEST -verify -x c++ %s
// RUN: %clang_cc1 -fsyntax-only -Wno-tautological-unsigned-zero-compare -verify -x c++ %s

unsigned uvalue(void);
signed int svalue(void);

#define macro(val) val

#ifdef __cplusplus
template<typename T>
void TFunc() {
  // Make sure that we do warn for normal variables in template functions !
  unsigned char c = svalue();
#ifdef TEST
  if (c < 0) // expected-warning {{comparison of unsigned expression < 0 is always false}}
      return;
#else
  if (c < 0)
      return;
#endif

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

  unsigned un = uvalue();

#ifdef TEST
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
#else
// expected-no-diagnostics
  if (un == 0)
      return 0;
  if (un != 0)
      return 0;
  if (un < 0)
      return 0;
  if (un <= 0)
      return 0;
  if (un > 0)
      return 0;
  if (un >= 0)
      return 0;

  if (0 == un)
      return 0;
  if (0 != un)
      return 0;
  if (0 < un)
      return 0;
  if (0 <= un)
      return 0;
  if (0 > un)
      return 0;
  if (0 >= un)
      return 0;

  if (un == 0UL)
      return 0;
  if (un != 0UL)
      return 0;
  if (un < 0UL)
      return 0;
  if (un <= 0UL)
      return 0;
  if (un > 0UL)
      return 0;
  if (un >= 0UL)
      return 0;

  if (0UL == un)
      return 0;
  if (0UL != un)
      return 0;
  if (0UL < un)
      return 0;
  if (0UL <= un)
      return 0;
  if (0UL > un)
      return 0;
  if (0UL >= un)
      return 0;
#endif


  signed int a = svalue();

#ifdef TEST
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
#else
// expected-no-diagnostics
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
  if (a < 0UL)
      return 0;
  if (a <= 0UL)
      return 0;
  if (a > 0UL)
      return 0;
  if (a >= 0UL)
      return 0;

  if (0UL == a)
      return 0;
  if (0UL != a)
      return 0;
  if (0UL < a)
      return 0;
  if (0UL <= a)
      return 0;
  if (0UL > a)
      return 0;
  if (0UL >= a)
      return 0;
#endif


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
