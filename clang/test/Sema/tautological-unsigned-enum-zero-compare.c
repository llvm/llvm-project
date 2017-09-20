// RUN: %clang_cc1 -fsyntax-only -DTEST -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wno-tautological-unsigned-enum-zero-compare -verify %s

int main() {
  enum A { A_foo, A_bar };
  enum A a;

#ifdef TEST
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
