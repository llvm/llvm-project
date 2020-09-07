// RUN: %clang_cc1 -fsyntax-only -Wignored-pragmas -verify %s

#pragma STDC FENV_ROUND ON   // expected-warning {{invalid or unsupported rounding mode}}

float func_01(int x, float y) {
  if (x)
    return y + 2;
  #pragma STDC FENV_ROUND FE_DOWNWARD // expected-error{{'#pragma STDC FENV_ROUND' can only appear at file scope or at the start of a compound statement}}
                                      // expected-warning@-1{{pragma STDC FENV_ROUND is not supported}}
  return x + y;
}
