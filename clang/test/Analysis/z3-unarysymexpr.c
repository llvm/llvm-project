// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s \
// RUN:  -analyzer-constraints=z3 
// RUN: %clang_analyze_cc1 -verify %s \
// RUN:  -analyzer-checker=core,debug.ExprInspection \
// RUN:  -analyzer-config crosscheck-with-z3=true

// REQUIRES: Z3
//
// Previously Z3 analysis crashed when it encountered an UnarySymExpr, validate
// that this no longer happens.
//

int negate(int x, int y) {
  if ( ~(x && y))
    return 0;
  return 1;
}

// Z3 is presented with a SymExpr like this : -((reg_$0<int a>) != 0) :
// from the Z3 refutation wrapper, that it attempts to convert to a
// SMTRefExpr, then crashes inside of Z3. The "not zero" portion
// of that expression is converted to the SMTRefExpr
// "(not (= reg_$1 #x00000000))", which is a boolean result then the
// "negative" operator (unary '-', UO_Minus) is attempted to be applied which
// then causes Z3 to crash. The accompanying patch just strips the negative
// operator before submitting to Z3 to avoid the crash.
//
// TODO: Find the root cause of this and fix it in symbol manager
//
void c();

int z3crash(int a, int b) {
  b = a || b;
  return (-b == a) / a; // expected-warning{{Division by zero [core.DivideZero]}}
}

// Floats are handled specifically, and differently in the Z3 refutation layer
// Just cover that code path
int z3_nocrash(float a, float b) {
  b = a || b;
  return (-b == a) / a;
}

int z3_crash2(int a) {
  int *d;
  if (-(&c && a))
    return *d; // expected-warning{{Dereference of undefined pointer value}}
  return 0;
}

// Refer to issue 165779
void z3_crash3(long a) {
  if (~-(5 && a)) {
    long *c;
    *c; // expected-warning{{Dereference of undefined pointer value (loaded from variable 'c')}}
  }
}
