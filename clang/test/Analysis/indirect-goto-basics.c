// RUN: %clang_analyze_cc1 -verify %s -analyzer-checker=core

// This file tests ExprEngine::processIndirectGoto and the class
// IndirectGotoNodeBuilder.

int goto_known_label_1(int x) {
  // In this example the ternary operator splits the state, the analyzer can
  // exactly follow the computed goto on both execution paths and validate that
  // we reach "33 / x" with a state where x is constrained to zero.
  void *target = x ? &&nonzero : &&zero;
  goto *target;
zero:
  return 33 / x; // expected-warning {{Division by zero}}
nonzero:
  return 66 / (x + 1);
}

int goto_known_label_2(int x) {
  // In this example the ternary operator splits the state, the analyzer can
  // exactly follow the computed goto on both execution paths and validate that
  // we do not reach "66 / x" with the state where x is constrained to zero.
  void *target = x ? &&nonzero : &&zero;
  goto *target;
zero:
  return 33 / (x + 1);
nonzero:
  return 66 / x;
}

void *select(int, void *, void *, void *);
int goto_symbolic(int x, int y) {
  // In this example the target of the indirect goto is a symbolic value so the
  // analyzer dispatches to all possible labels and we get the zero division
  // errors at all of them.
  void *target = select(x, &&first, &&second, &&third);
  if (y)
    return 41;
  goto *target;
first:
  return 33 / y; // expected-warning {{Division by zero}}
second:
  return 66 / y; // expected-warning {{Division by zero}}
third:
  return 123 / y; // expected-warning {{Division by zero}}
}

int goto_nullpointer(int x, int y) {
  // In this example the target of the indirect goto is a loc::ConcreteInt (a
  // null pointer), so the analyzer doesn't dispatch anywhere.
  // FIXME: The analyzer should report that nullptr (or some other concrete
  // value) was passed to an indirect goto.
  void *target = (void *)0;
  (void)&&first;
  (void)&&second;
  (void)&&third;
  if (y)
    return 41;
  goto *target;
first:
  return 33 / y;
second:
  return 66 / y;
third:
  return 123 / y;
}

int goto_undefined(int x, int y) {
  // In this example the target of the indirect goto is an uninitialized
  // pointer, so the analyzer doesn't dispatch anywhere.
  // FIXME: The analyzer should report that an uninitialized value was passed
  // to an indirect goto.
  void *target;
  (void)&&first;
  (void)&&second;
  (void)&&third;
  if (y)
    return 41;
  goto *target;
first:
  return 33 / y;
second:
  return 66 / y;
third:
  return 123 / y;
}
