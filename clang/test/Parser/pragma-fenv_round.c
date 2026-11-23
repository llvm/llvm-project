// RUN: %clang_cc1 -fexperimental-strict-floating-point -fsyntax-only -Wignored-pragmas -verify %s

#pragma STDC FENV_ROUND ON   // expected-warning {{invalid or unsupported rounding mode}}

float func_01(int x, float y) {
  if (x)
    return y + 2;
  #pragma STDC FENV_ROUND FE_DOWNWARD // expected-error{{'#pragma STDC FENV_ROUND' can only appear at file scope or at the start of a compound statement}}
                                      // expected-warning@-1{{pragma STDC FENV_ROUND is not supported}}
  return x + y;
}

// Rounding direction names are macros in <fenv.h>; STDC pragmas are not macro-expanded.
#define FE_DOWNWARD 0x400
#define FE_UPWARD	0x800
#pragma STDC FENV_ROUND FE_DOWNWARD // expected-warning{{expected identifier in '#pragma FENV_ROUND' - ignored}}

#define MY_ROUND FE_UPWARD
#pragma STDC FENV_ROUND MY_ROUND // expected-warning{{expected identifier in '#pragma FENV_ROUND' - ignored}}
                                 // expected-error@-1{{expected identifier or '('}}
