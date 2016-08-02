// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.ExprInspection -verify %s
// expected-no-diagnostics
struct X {
  int *p;
  int zero;
  void foo () {
    reset(p - 1);
  }
  void reset(int *in) {
    while (in != p) // Loop must be entered.
      zero = 1;
  }
};

int test (int *in) {
  X littleX;
  littleX.zero = 0;
  littleX.p = in;
  littleX.foo();
  return 5/littleX.zero; // no-warning
}

