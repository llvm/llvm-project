// RUN: %clang_analyze_cc1 \
// RUN:  -analyzer-checker=core -analyzer-config \
// RUN:   silence-checkers=core.DivideZero \
// RUN:  -verify %s

void test_disable_core_div_by_zero() {
  (void)(1 / 0);
  // expected-warning@-1 {{division by zero is undefined}}
  // no-warning: 'Division by zero'
}

void test_disable_null_deref(int *p) {
  if (p)
    return;

  int x = p[0];
  // expected-warning@-1 {{Array access (from variable 'p') results in a null pointer dereference}}
}
