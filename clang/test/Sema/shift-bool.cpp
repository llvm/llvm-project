// RUN: %clang_cc1 -fsyntax-only -Wshift-bool -verify %s

void t() {
  int x = 10;
  bool y = true;
  int z = 1;

  bool a = y << x;
  bool b = y >> x; // expected-warning {{right shifting a 'bool' implicitly converts it to 'int'}}

  bool c = false << x;
  bool d = false >> x; // expected-warning {{right shifting a 'bool' implicitly converts it to 'int'}}

  bool e = y << 1;
  bool f = y >> 1; // expected-warning {{right shifting a 'bool' implicitly converts it to 'int'}}

  bool g = y << -1; // expected-warning {{shift count is negative}}
  bool h = y >> -1; // expected-warning {{right shifting a 'bool' implicitly converts it to 'int'}} \
                    // expected-warning {{shift count is negative}}

  bool i = y << 0;
  bool j = y >> 0; // expected-warning {{right shifting a 'bool' implicitly converts it to 'int'}}

  bool k = (x < z) >> 1; // expected-warning {{right shifting a 'bool' implicitly converts it to 'int'}}

  if ((y << 1) != 0) { }
  if ((y >> 1) != 0) { } // expected-warning {{right shifting a 'bool' implicitly converts it to 'int'}}
}
