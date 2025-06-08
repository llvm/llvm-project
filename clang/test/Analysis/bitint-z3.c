// RUN: %clang_cc1 -analyze -analyzer-checker=core -w -DNO_CROSSCHECK -verify %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core -w -analyzer-config crosscheck-with-z3=true -verify %s
// REQUIRES: z3

// The SMTConv layer did not comprehend _BitInt types (because this was an
// evolved feature) and was crashing due to needed updates in 2 places.
// Analysis is expected to find these cases using _BitInt without crashing.

_BitInt(35) a;
int b;
void c() {
  int d;
  if (a)
    b = d; // expected-warning{{Assigned value is uninitialized [core.uninitialized.Assign]}}
}

int *d;
_BitInt(3) e;
void f() {
  int g;
  d = &g;
  e ?: 0; // expected-warning{{Address of stack memory associated with local variable 'g' is still referred to by the global variable 'd' upon returning to the caller.  This will be a dangling reference [core.StackAddressEscape]}}
}
