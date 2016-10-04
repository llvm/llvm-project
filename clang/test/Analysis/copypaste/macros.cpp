// RUN: %clang_cc1 -analyze -std=c++11 -analyzer-checker=alpha.clone.CloneChecker -verify %s

// Tests that macros and non-macro clones aren't mixed into the same hash
// group. This is currently necessary as all clones in a hash group need
// to have the same complexity value. Macros have smaller complexity values
// and need to be in their own hash group.

int foo(int a) { // expected-warning{{Detected code clone.}}
  a = a + 1;
  a = a + 1 / 1;
  a = a + 1 + 1 + 1;
  a = a + 1 - 1 + 1 + 1;
  a = a + 1 * 1 + 1 + 1 + 1;
  a = a + 1 / 1 + 1 + 1 + 1;
  return a;
}

int fooClone(int a) { // expected-note{{Related code clone is here.}}
  a = a + 1;
  a = a + 1 / 1;
  a = a + 1 + 1 + 1;
  a = a + 1 - 1 + 1 + 1;
  a = a + 1 * 1 + 1 + 1 + 1;
  a = a + 1 / 1 + 1 + 1 + 1;
  return a;
}

// Below is the same AST as above but this time generated with macros. The
// clones below should land in their own hash group for the reasons given above.

#define ASSIGN(T, V) T = T + V

int macro(int a) { // expected-warning{{Detected code clone.}}
  ASSIGN(a, 1);
  ASSIGN(a, 1 / 1);
  ASSIGN(a, 1 + 1 + 1);
  ASSIGN(a, 1 - 1 + 1 + 1);
  ASSIGN(a, 1 * 1 + 1 + 1 + 1);
  ASSIGN(a, 1 / 1 + 1 + 1 + 1);
  return a;
}

int macroClone(int a) { // expected-note{{Related code clone is here.}}
  ASSIGN(a, 1);
  ASSIGN(a, 1 / 1);
  ASSIGN(a, 1 + 1 + 1);
  ASSIGN(a, 1 - 1 + 1 + 1);
  ASSIGN(a, 1 * 1 + 1 + 1 + 1);
  ASSIGN(a, 1 / 1 + 1 + 1 + 1);
  return a;
}

// FIXME: Macros with empty definitions in the AST are currently ignored.

#define EMPTY

int fooFalsePositiveClone(int a) { // expected-note{{Related code clone is here.}}
  a = EMPTY a + 1;
  a = a + 1 / 1;
  a = a + 1 + 1 + 1;
  a = a + 1 - 1 + 1 + 1;
  a = a + 1 * 1 + 1 + 1 + 1;
  a = a + 1 / 1 + 1 + 1 + 1;
  return a;
}


