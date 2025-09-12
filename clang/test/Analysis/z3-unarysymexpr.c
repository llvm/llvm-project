// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s \
// RUN:  -analyzer-constraints=z3 

// REQUIRES: Z3
//
// Previously Z3 analysis crashed when it encountered an UnarySymExpr, validate
// that this no longer happens.
//

// expected-no-diagnostics
int negate(int x, int y) {
  if ( ~(x && y))
    return 0;
  return 1;
}

void c();
void case004(int *a, int *b) {
  void *e;
  b != a;
  c(e); // expected-warning{{1st function call argument is an uninitialized value}}
}

void z3crash(int a, int b) {
  b = a || b;
  (-b == a) / a; // expected-warning{{expression result unused}}
                 // expected-warning@-1{{Division by zero [core.DivideZero]}}
}

void z3_nocrash(float a, float b) {
  b = a || b;
  (-b == a) / a; // expected-warning{{expression result unused}}
}

void z3_crash2(int a) {
  if (-(&c && a)) {
    int *d;
    *d; // expected-warning{{Dereference of undefined pointer value}}
        // expected-warning@-1{{expression result unused}}
  }
}
