// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -verify

void clang_analyzer_warnIfReached();
void clang_analyzer_eval();

void test_simplification_to_concrete_int_infeasible(int b, int c) {
  if (c + b != 0)     // c + b == 0
    return;
  if (b != 1)         // b == 1  --> c + 1 == 0
    return;
  if (c != 1)         // c == 1  --> 1 + 1 == 0 contradiction
    return;
  clang_analyzer_warnIfReached(); // no-warning

  // Keep the symbols and the constraints! alive.
  (void)(b * c);
  return;
}

void test_simplification_to_concrete_int_feasible(int b, int c) {
  if (c + b != 0)
    return;
                      // c + b == 0
  if (b != 1)
    return;
                      // b == 1   -->  c + 1 == 0
  if (c != -1)
    return;
                      // c == -1  --> -1 + 1 == 0 OK
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  clang_analyzer_eval(c == -1);   // expected-warning{{TRUE}}

  // Keep the symbols and the constraints! alive.
  (void)(b * c);
  return;
}
