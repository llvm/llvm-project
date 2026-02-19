// RUN: %clang_analyze_cc1 -verify %s -analyzer-checker=core

// This file tests ExprEngine::processSwitch and the class SwitchNodeBuilder.

int switch_simple(int x) {
  // Validate that switch behaves as expected in a very simple situation.
  switch (x) {
    case 1:
      return 13 / x;
    case 0:
      return 14 / x; // expected-warning {{Division by zero}}
    case 2:
      return 15 / x;
    default:
      return 67 / (x - x); // expected-warning {{Division by zero}}
  }
}

int switch_default(int x) {
  // Validate that the default case is evaluated after excluding the other
  // cases -- even if it appears first.
  if (x > 2 || x < 0)
    return 0;
  switch (x) {
    default:
      return 16 / x; // expected-warning {{Division by zero}}
    case 1:
      return 13 / x;
    case 2:
      return 15 / x;
  }
}

int switch_unreachable_default(int x) {
  // Validate that the default case is not evaluated if it is infeasible.
  int zero = 0;
  if (x > 2 || x < 0)
    return 0;
  switch (x) {
    default:
      return 16 / zero; // no-warning
    case 0:
      return 456;
    case 1:
      return 13 / x;
    case 2:
      return 15 / x;
  }
}

enum Color {Red, Green, Blue};

int switch_all_enum_cases_covered(enum Color x) {
  // Validate that the default case is not evaluated if the switch is over an
  // enum value and all enumerators appear as 'case's.
  int zero = 0;
  switch (x) {
    default:
      return 16 / zero; // no-warning
    case Red:
    case Green:
      return 2;
    case Blue:
      return 3;
  }
}

int switch_all_feasible_enum_cases_covered(enum Color x) {
  // Highlight a shortcoming of enum/switch handling: here the 'case's cover
  // all the enumerators that could appear in the symbolic value 'x', but the
  // default is still executed.
  // FIXME: The default branch shouldn't be executed here.
  int zero = 0;

  if (x == Red)
    return 1;
  switch (x) {
    default:
      return 16 / zero; // expected-warning {{Division by zero}}
    case Green:
      return 2;
    case Blue:
      return 3;
  }
}

int switch_no_compound_stmt(int x) {
  // Validate that the engine can follow the switch statement even if there is
  // no compound statement around the cases. (Yes, this is valid, although
  // not very practical.)
  switch (x) case 1: case 0: return 16 / x; // expected-warning {{Division by zero}}

  return 0;
}

int switch_with_case_range(int x) {
  // Validate that the GNU case range extension is properly handled.
  switch (x) {
    case 5:
      return 55 / x;
    case 2 ... 4:
      return 3;
    case 0 ... 1:
      return 44 / x; // no-warning: there is no state split between 0 and 1
    default:
      if (x)
        return 8;
      return 45 / x; // no-warning: x cannot be 0 on the default branch
  }
}
