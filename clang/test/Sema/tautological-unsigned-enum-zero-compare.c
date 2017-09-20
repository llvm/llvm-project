// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -fsyntax-only -DALL_WARN -verify %s
// RUN: %clang_cc1 -triple=x86_64-pc-win32 -fsyntax-only -DSIGN_WARN -verify %s
// RUN: %clang_cc1 -triple=x86_64-pc-win32 -fsyntax-only -Wno-tautological-unsigned-enum-zero-compare -verify %s

// Okay, this is where it gets complicated.
// Then default enum sigdness is target-specific.
// On windows, it is signed by default. We do not want to warn in that case.

int main() {
  enum A { A_foo, A_bar };
  enum A a;

#ifdef ALL_WARN
  if (a < 0) // expected-warning {{comparison of unsigned enum expression < 0 is always false}}
    return 0;
  if (a >= 0) // expected-warning {{comparison of unsigned enum expression >= 0 is always true}}
    return 0;
  if (0 <= a) // expected-warning {{comparison of 0 <= unsigned enum expression is always true}}
    return 0;
  if (0 > a) // expected-warning {{comparison of 0 > unsigned enum expression is always false}}
    return 0;
  if (a < 0U) // expected-warning {{comparison of unsigned enum expression < 0 is always false}}
    return 0;
  if (a >= 0U) // expected-warning {{comparison of unsigned enum expression >= 0 is always true}}
    return 0;
  if (0U <= a) // expected-warning {{comparison of 0 <= unsigned enum expression is always true}}
    return 0;
  if (0U > a) // expected-warning {{comparison of 0 > unsigned enum expression is always false}}
    return 0;
#elif defined(SIGN_WARN)
  if (a < 0) // ok
    return 0;
  if (a >= 0) // ok
    return 0;
  if (0 <= a) // ok
    return 0;
  if (0 > a) // ok
    return 0;
  if (a < 0U) // expected-warning {{comparison of unsigned enum expression < 0 is always false}}
    return 0;
  if (a >= 0U) // expected-warning {{comparison of unsigned enum expression >= 0 is always true}}
    return 0;
  if (0U <= a) // expected-warning {{comparison of 0 <= unsigned enum expression is always true}}
    return 0;
  if (0U > a) // expected-warning {{comparison of 0 > unsigned enum expression is always false}}
    return 0;
#else
  // expected-no-diagnostics
  if (a < 0)
    return 0;
  if (a >= 0)
    return 0;
  if (0 <= a)
    return 0;
  if (0 > a)
    return 0;
  if (a < 0U)
    return 0;
  if (a >= 0U)
    return 0;
  if (0U <= a)
    return 0;
  if (0U > a)
    return 0;
#endif

  return 1;
}
