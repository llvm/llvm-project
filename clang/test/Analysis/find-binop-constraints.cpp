// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -verify

void clang_analyzer_eval(bool);
void clang_analyzer_warnIfReached();

int test_legacy_behavior(int x, int y) {
  if (y != 0)
    return 0;
  if (x + y != 0)
    return 0;
  clang_analyzer_eval(x + y == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(y == 0);     // expected-warning{{TRUE}}
  return y / (x + y);              // expected-warning{{Division by zero}}
}

int test_rhs_further_constrained(int x, int y) {
  if (x + y != 0)
    return 0;
  if (y != 0)
    return 0;
  clang_analyzer_eval(x + y == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(y == 0);     // expected-warning{{TRUE}}
  return 0;
}

int test_lhs_further_constrained(int x, int y) {
  if (x + y != 0)
    return 0;
  if (x != 0)
    return 0;
  clang_analyzer_eval(x + y == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(x == 0);     // expected-warning{{TRUE}}
  return 0;
}

int test_lhs_and_rhs_further_constrained(int x, int y) {
  if (x % y != 1)
    return 0;
  if (x != 1)
    return 0;
  if (y != 2)
    return 0;
  clang_analyzer_eval(x % y == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(y == 2);     // expected-warning{{TRUE}}
  return 0;
}

int test_commutativity(int x, int y) {
  if (x + y != 0)
    return 0;
  if (y != 0)
    return 0;
  clang_analyzer_eval(y + x == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(y == 0);     // expected-warning{{TRUE}}
  return 0;
}

int test_binop_when_height_is_2_r(int a, int x, int y, int z) {
  switch (a) {
  case 1: {
    if (x + y + z != 0)
      return 0;
    if (z != 0)
      return 0;
    clang_analyzer_eval(x + y + z == 0); // expected-warning{{TRUE}}
    clang_analyzer_eval(z == 0);         // expected-warning{{TRUE}}
    break;
  }
  case 2: {
    if (x + y + z != 0)
      return 0;
    if (y != 0)
      return 0;
    clang_analyzer_eval(x + y + z == 0); // expected-warning{{TRUE}}
    clang_analyzer_eval(y == 0);         // expected-warning{{TRUE}}
    break;
  }
  case 3: {
    if (x + y + z != 0)
      return 0;
    if (x != 0)
      return 0;
    clang_analyzer_eval(x + y + z == 0); // expected-warning{{TRUE}}
    clang_analyzer_eval(x == 0);         // expected-warning{{TRUE}}
    break;
  }
  case 4: {
    if (x + y + z != 0)
      return 0;
    if (x + y != 0)
      return 0;
    clang_analyzer_eval(x + y + z == 0); // expected-warning{{TRUE}}
    clang_analyzer_eval(x + y == 0);     // expected-warning{{TRUE}}
    break;
  }
  case 5: {
    if (z != 0)
      return 0;
    if (x + y + z != 0)
      return 0;
    clang_analyzer_eval(x + y + z == 0); // expected-warning{{TRUE}}
    if (y != 0)
      return 0;
    clang_analyzer_eval(y == 0);         // expected-warning{{TRUE}}
    clang_analyzer_eval(z == 0);         // expected-warning{{TRUE}}
    clang_analyzer_eval(x + y + z == 0); // expected-warning{{TRUE}}
    break;
  }

  }
  return 0;
}

void test_equivalence_classes_are_updated(int a, int b, int c, int d) {
  if (a + b != c)
    return;
  if (a != d)
    return;
  if (b != 0)
    return;
  clang_analyzer_eval(c == d); // expected-warning{{TRUE}}
  // Keep the symbols and the constraints! alive.
  (void)(a * b * c * d);
  return;
}

void test_contradiction(int a, int b, int c, int d) {
  if (a + b != c)
    return;
  if (a == c)
    return;
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}

  // Bring in the contradiction.
  if (b != 0)
    return;
  clang_analyzer_warnIfReached(); // no-warning, i.e. UNREACHABLE
  // Keep the symbols and the constraints! alive.
  (void)(a * b * c * d);
  return;
}

void test_deferred_contradiction(int e0, int b0, int b1) {

  int e1 = e0 - b0; // e1 is bound to (reg_$0<int e0>) - (reg_$1<int b0>)
  (void)(b0 == 2);  // bifurcate

  int e2 = e1 - b1;
  if (e2 > 0) { // b1 != e1
    clang_analyzer_warnIfReached();   // expected-warning{{REACHABLE}}
    // Here, e1 is still bound to (reg_$0<int e0>) - (reg_$1<int b0>) but we
    // should be able to simplify it to (reg_$0<int e0>) - 2 and thus realize
    // the contradiction.
    if (b1 == e1) {
      clang_analyzer_warnIfReached(); // no-warning, i.e. UNREACHABLE
      (void)(b0 * b1 * e0 * e1 * e2);
    }
  }
}
