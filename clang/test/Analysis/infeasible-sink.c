// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -verify

// Here we test that if it turns out that the parent state is infeasible then
// both children States (more precisely the ExplodedNodes) are marked as a
// Sink.
// We rely on existing defects of the underlying constraint solver. However,
// in the future we might strengthen the solver to discover the infeasibility
// right when we create the parent state. At that point some of these tests
// will fail, and either we shall find another solver weakness to have the test
// case functioning, or we shall simply remove that.

void clang_analyzer_warnIfReached();
void clang_analyzer_eval(int);

void test1(int x) {
  if (x * x != 4)
    return;
  if (x < 0 || x > 1)
    return;

  // { x^2 == 4 and x:[0,1] }
  // This state is already infeasible.

  // Perfectly constraining 'x' will trigger constant folding,
  // when we realize we were already infeasible.
  // The same happens for the 'else' branch.
  if (x == 0) {
    clang_analyzer_warnIfReached(); // no-warning
  } else {
    clang_analyzer_warnIfReached(); // no-warning
  }
  clang_analyzer_warnIfReached(); // no-warning
  (void)x;
}

int a, b, c, d, e;
void test2(void) {

  if (a == 0)
    return;

  if (e != c)
    return;

  d = e - c;
  b = d;
  a -= d;

  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}

  if (a != 0)
    return;

  clang_analyzer_warnIfReached(); // no-warning: Unreachable due to contradiction.
}
