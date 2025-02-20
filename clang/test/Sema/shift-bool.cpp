// RUN: %clang_cc1 -fsyntax-only -Wshift-bool -verify %s

void t() {
  int x = 10;
  bool y = true;

  bool a = y >> x;     // expected-warning {{right shifting a `bool` implicitly converts it to 'int'}}
  bool b = false >> x; // expected-warning {{right shifting a `bool` implicitly converts it to 'int'}}
  bool c = y >> 5;     // expected-warning {{right shifting a `bool` implicitly converts it to 'int'}}
  bool d = y >> -1;    // expected-warning {{right shifting a `bool` implicitly converts it to 'int'}}
  bool e = y >> 0;     // expected-warning {{right shifting a `bool` implicitly converts it to 'int'}}
}
