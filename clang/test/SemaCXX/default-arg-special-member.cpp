// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -Wno-default-arg-special-member -Werror -fsyntax-only %s

class foo {
  foo(foo&, int); // expected-note {{was not a special member function}}
  foo(int); // expected-note {{was not a special member function}}
  foo(const foo&); // expected-note {{was a copy constructor}}
};

foo::foo(foo&, int = 0) { } // expected-warning {{makes this constructor a copy constructor}}
foo::foo(int = 0) { } // expected-warning {{makes this constructor a default constructor}}
foo::foo(const foo& = 0) { } //expected-warning {{makes this constructor a default constructor}}
