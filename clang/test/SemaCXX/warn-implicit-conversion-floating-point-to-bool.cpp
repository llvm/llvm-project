// RUN: %clang_cc1 -verify -fsyntax-only %s

float foof(float x);
double food(double x);
void foo(bool b, float f);

void bar() {

  float c = 1.7;
  bool b = c;

  double e = 1.7;
  b = e;

  b = foof(4.0);

  b = foof(c < 1); // expected-warning {{implicit conversion turns floating-point number into bool: 'float' to 'bool'}}

  b = food(e < 2); // expected-warning {{implicit conversion turns floating-point number into bool: 'double' to 'bool'}}

  foo(c, b);    // expected-warning {{implicit conversion turns floating-point number into bool: 'float' to 'bool'}}
  foo(c, c);

}
