// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -verify

// Here we test whether the SValBuilder is capable to simplify existing
// compound SVals (where there are at leaset 3 symbols in the tree) based on
// newly added constraints.

void clang_analyzer_eval(bool);
void clang_analyzer_warnIfReached();

void test_left_tree_constrained(int x, int y, int z) {
  if (x + y + z != 0)
    return;
  if (x + y != 0)
    return;
  clang_analyzer_eval(x + y + z == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(x + y == 0);     // expected-warning{{TRUE}}
  clang_analyzer_eval(z == 0);         // expected-warning{{TRUE}}
  x = y = z = 1;
  return;
}

void test_right_tree_constrained(int x, int y, int z) {
  if (x + y * z != 0)
    return;
  if (y * z != 0)
    return;
  clang_analyzer_eval(x + y * z == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(y * z == 0);     // expected-warning{{TRUE}}
  clang_analyzer_eval(x == 0);         // expected-warning{{TRUE}}
  return;
}

void test_left_tree_constrained_minus(int x, int y, int z) {
  if (x - y - z != 0)
    return;
  if (x - y != 0)
    return;
  clang_analyzer_eval(x - y - z == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(x - y == 0);     // expected-warning{{TRUE}}
  clang_analyzer_eval(z == 0);         // expected-warning{{TRUE}}
  x = y = z = 1;
  return;
}

void test_SymInt_constrained(int x, int y, int z) {
  if (x * y * z != 4)
    return;
  if (z != 2)
    return;
  if (x * y == 3) {
    clang_analyzer_warnIfReached();     // no-warning
    return;
  }
  (void)(x * y * z);
}

void test_SValBuilder_simplifies_IntSym(int x, int y, int z) {
  // Most IntSym BinOps are transformed to SymInt in SimpleSValBuilder.
  // Division is one exception.
  x = 77 / (y + z);
  if (y + z != 1)
    return;
  clang_analyzer_eval(x == 77);         // expected-warning{{TRUE}}
  (void)(x * y * z);
}

void recurring_symbol(int b) {
  if (b * b != b)
    if ((b * b) * b * b != (b * b) * b)
      if (b * b == 1)                   // no-crash (assert should not fire)
        clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}
