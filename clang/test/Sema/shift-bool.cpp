// RUN: %clang_cc1 -fsyntax-only -Wshift-bool -verify %s

void t() {
  int x = 10;
  bool y = true;

  bool a = y << x; // expected-warning {{left shifting a `bool` implicitly converts it to 'int'}}
  bool b = y >> x; // expected-warning {{right shifting a `bool` implicitly converts it to 'int'}}

  bool c = false << x; // expected-warning {{left shifting a `bool` implicitly converts it to 'int'}}
  bool d = false >> x; // expected-warning {{right shifting a `bool` implicitly converts it to 'int'}}

  bool e = y << 5; // expected-warning {{left shifting a `bool` implicitly converts it to 'int'}}
  bool f = y >> 5; // expected-warning {{right shifting a `bool` implicitly converts it to 'int'}}

  bool g = y << -1; // expected-warning {{left shifting a `bool` implicitly converts it to 'int'}}
  bool h = y >> -1; // expected-warning {{right shifting a `bool` implicitly converts it to 'int'}}

  bool i = y << 0; // expected-warning {{left shifting a `bool` implicitly converts it to 'int'}}
  bool j = y >> 0; // expected-warning {{right shifting a `bool` implicitly converts it to 'int'}}
}
