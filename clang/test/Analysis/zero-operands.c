// RUN: %clang_analyze_cc1 -analyzer-checker=core \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -verify %s

void clang_analyzer_dump(int);

void test_0_multiplier1(int x, int y) {
  int a = x < 0; // Eagerly bifurcate.
  clang_analyzer_dump(a);
  // expected-warning@-1{{0 S32b}}
  // expected-warning@-2{{1 S32b}}

  int b = a * y;
  clang_analyzer_dump(b);
  // expected-warning@-1{{0 S32b}}
  // expected-warning-re@-2{{reg_${{[[:digit:]]+}}<int y>}}
}

void test_0_multiplier2(int x, int y) {
  int a = x < 0; // Eagerly bifurcate.
  clang_analyzer_dump(a);
  // expected-warning@-1{{0 S32b}}
  // expected-warning@-2{{1 S32b}}

  int b = y * a;
  clang_analyzer_dump(b);
  // expected-warning@-1{{0 S32b}}
  // expected-warning-re@-2{{reg_${{[[:digit:]]+}}<int y>}}
}

void test_0_modulo(int x, int y) {
  int a = x < 0; // Eagerly bifurcate.
  clang_analyzer_dump(a);
  // expected-warning@-1{{0 S32b}}
  // expected-warning@-2{{1 S32b}}

  int b = a % y;
  clang_analyzer_dump(b);
  // expected-warning@-1{{0 S32b}}
  // expected-warning-re@-2{{1 % (reg_${{[[:digit:]]+}}<int y>)}}
}

void test_0_divisible(int x, int y) {
  int a = x < 0; // Eagerly bifurcate.
  clang_analyzer_dump(a);
  // expected-warning@-1{{0 S32b}}
  // expected-warning@-2{{1 S32b}}

  int b = a / y;
  clang_analyzer_dump(b);
  // expected-warning@-1{{0 S32b}}
  // expected-warning-re@-2{{1 / (reg_${{[[:digit:]]+}}<int y>)}}
}
