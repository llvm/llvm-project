// RUN: %clang_cc1 -fsyntax-only -DTEST -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wno-tautological-unsigned-zero-compare -verify %s

unsigned value(void);

int main() {
  unsigned un = value();

#ifdef TEST
  if (un < 0) // expected-warning {{comparison of unsigned expression < 0 is always false}}
    return 0;
  if (un >= 0) // expected-warning {{comparison of unsigned expression >= 0 is always true}}
    return 0;
  if (0 <= un) // expected-warning {{comparison of 0 <= unsigned expression is always true}}
    return 0;
  if (0 > un) // expected-warning {{comparison of 0 > unsigned expression is always false}}
    return 0;
  if (un < 0U) // expected-warning {{comparison of unsigned expression < 0 is always false}}
    return 0;
  if (un >= 0U) // expected-warning {{comparison of unsigned expression >= 0 is always true}}
    return 0;
  if (0U <= un) // expected-warning {{comparison of 0 <= unsigned expression is always true}}
    return 0;
  if (0U > un) // expected-warning {{comparison of 0 > unsigned expression is always false}}
    return 0;
#else
// expected-no-diagnostics
  if (un < 0)
    return 0;
  if (un >= 0)
    return 0;
  if (0 <= un)
    return 0;
  if (0 > un)
    return 0;
  if (un < 0U)
    return 0;
  if (un >= 0U)
    return 0;
  if (0U <= un)
    return 0;
  if (0U > un)
    return 0;
#endif

  return 1;
}
