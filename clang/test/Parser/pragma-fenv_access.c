// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -ffp-exception-behavior=strict -DSTRICT -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -x c++ -DCPP -DSTRICT -ffp-exception-behavior=strict -fsyntax-only -verify %s
#ifdef CPP
#define CONST constexpr
#else
#define CONST const
#endif

#pragma STDC FENV_ACCESS IN_BETWEEN   // expected-warning {{expected 'ON' or 'OFF' or 'DEFAULT' in pragma}}

#pragma STDC FENV_ACCESS OFF

float func_04(int x, float y) {
  if (x)
    return y + 2;
  #pragma STDC FENV_ACCESS ON // expected-error{{'#pragma STDC FENV_ACCESS' can only appear at file scope or at the start of a compound statement}}
  return x + y;
}

#pragma STDC FENV_ACCESS ON
int main(void) {
  CONST float one = 1.0F ;
  CONST float three = 3.0F ;
  CONST float four = 4.0F ;
  CONST float frac_ok = one/four;
#if !defined(CPP)
//expected-note@+2 {{declared here}}
#endif
  CONST float frac = one/three;
  CONST double d = one;
  CONST int not_too_big = 255;
  CONST float fnot_too_big = not_too_big;
  CONST int too_big = 0x7ffffff0;
#if defined(CPP)
//expected-warning@+2{{implicit conversion from 'const int' to 'const float' changes value from 2147483632 to 2147483648}}
#endif
  CONST float fbig = too_big; // inexact
#if !defined(CPP)
#define static_assert _Static_assert
#endif
enum {
  e1 = (int)one, e3 = (int)three, e4 = (int)four, e_four_quarters = (int)(frac_ok * 4)
};
static_assert(e1 == 1  && e3 == 3 && e4 == 4 && e_four_quarters == 1, "");
enum {
#if !defined(CPP)
// expected-error@+2 {{not an integer constant expression}} expected-note@+2 {{is not a constant expression}}
#endif
  e_three_thirds = (int)(frac * 3)
};
  if (one <= four)  return 0;
  return -1;
}
