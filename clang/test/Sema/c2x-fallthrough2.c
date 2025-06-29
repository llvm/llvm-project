// RUN: %clang_cc1 -fsyntax-only -std=c2x -verify %s

// This is the latest version of fallthrough that we support.
_Static_assert(__has_c_attribute(fallthrough) == 201910L);

void f(int n) {
  switch (n) {
  case 8:
    [[fallthrough]]; // expected-error {{does not directly precede switch label}}
    goto label;
  label:
  case 9:
    n += 1;
  case 10: // no warning, -Wimplicit-fallthrough is not enabled in this test, and does not need to
           // be enabled for these diagnostics to be produced.
    break;
  }
}
