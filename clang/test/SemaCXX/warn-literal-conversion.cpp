// RUN: %clang_cc1 -fsyntax-only -Wliteral-conversion -verify %s

void foo(int y);

// Warn when a literal float or double is assigned or bound to an integer.
void test0() {
  // Float
  int y0 = 1.2222F; // expected-warning {{implicit conversion from 'float' to 'int' changes value from 1.2222 to 1}}
  int y1 = (1.2222F); // expected-warning {{implicit conversion from 'float' to 'int' changes value from 1.2222 to 1}}
  int y2 = (((1.2222F))); // expected-warning {{implicit conversion from 'float' to 'int' changes value from 1.2222 to 1}}
  int y3 = 12E-1F; // expected-warning {{implicit conversion from 'float' to 'int' changes value from 1.2 to 1}}
  int y4 = 1.23E1F; // expected-warning {{implicit conversion from 'float' to 'int' changes value from 12.3 to 12}}
  // Double
  int y5 = 1.2222; // expected-warning {{implicit conversion from 'double' to 'int' changes value from 1.2222 to 1}}
  int y6 = 12E-1; // expected-warning {{implicit conversion from 'double' to 'int' changes value from 1.2 to 1}}
  int y7 = 1.23E1; // expected-warning {{implicit conversion from 'double' to 'int' changes value from 12.3 to 12}}
  int y8 = (1.23E1); // expected-warning {{implicit conversion from 'double' to 'int' changes value from 12.3 to 12}}

  // Test assignment to an existing variable.
  y8 = 2.22F; // expected-warning {{implicit conversion from 'float' to 'int' changes value from 2.22 to 2}}

  // Test direct initialization.
  int y9(1.23F); // expected-warning {{implicit conversion from 'float' to 'int' changes value from 1.23 to 1}}

  // Test passing a literal floating-point value to a function that takes an integer.
  foo(1.2F); // expected-warning {{implicit conversion from 'float' to 'int' changes value from 1.2 to 1}}

  int y10 = -1.2F;  // expected-warning {{implicit conversion from 'float' to 'int' changes value from -1.2 to -1}}

  // -Wliteral-conversion does NOT catch const values.
  // (-Wconversion DOES catch them.)
  static const float sales_tax_rate = .095F;
  int z = sales_tax_rate;
  foo(sales_tax_rate);

  // Expressions, such as those that indicate rounding-down, should NOT produce warnings.
  int x = 24 * 0.5;
  int y = (24*60*60) * 0.25;
  int pennies = 123.45 * 100;
}

// Similarly, test floating point conversion to bool. Only float values of zero
// are converted to false; everything else is converted to true.
void test1() {
  bool b1 = 0.99f; // expected-warning {{implicit conversion from 'float' to 'bool' changes value from 0.99 to true}}
  bool b2 = 0.99; // expected-warning {{implicit conversion from 'double' to 'bool' changes value from 0.99 to true}}
  // These do not warn because they can be directly converted to integral
  // values.
  bool b3 = 0.0f;
  bool b4 = 0.0;
}
