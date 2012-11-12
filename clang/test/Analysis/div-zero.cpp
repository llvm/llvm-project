// RUN: %clang_cc1 -analyze -analyzer-checker=core.DivideZero -verify %s

int fooPR10616 (int qX ) {
  int a, c, d;

  d = (qX-1);
  while ( d != 0 ) {
    d = c - (c/d) * d;
  }

  return (a % (qX-1)); // expected-warning {{Division by zero}}

}
