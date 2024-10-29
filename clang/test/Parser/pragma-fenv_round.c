// RUN: %clang_cc1 -fexperimental-strict-floating-point -fsyntax-only -Wignored-pragmas -verify %s

#pragma STDC FENV_ROUND      // expected-error{{expected rounding mode}}
#pragma STDC FENV_ROUND ON   // expected-error{{invalid or unsupported rounding mode}}
#pragma STDC FENV_ROUND FE_DYNAMIC 0   // expected-warning{{extra tokens at end of #pragma STDC FENV_ROUND directive}}

float func_01(int x, float y) {
  if (x)
    return y + 2;
  #pragma STDC FENV_ROUND FE_DOWNWARD // expected-error{{'#pragma STDC FENV_ROUND' can only appear at file scope or at the start of a compound statement}}
  return x + y;
}
