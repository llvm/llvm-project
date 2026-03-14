// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

int i = delete; // expected-error{{only functions}}
int j = default; // expected-error{{special member functions}}

int f() = delete, g; // expected-error{{'= delete' is a function definition}}
int o, p() = delete; // expected-error{{'= delete' is a function definition}}

int q() = default, r; // expected-error{{only special member functions}} \
                      // expected-error{{'= default' is a function definition}}
int s, t() = default; // expected-error{{'= default' is a function definition}}

struct foo {
  foo() = default;
  ~foo() = delete;
  void bar() = delete;
};

void baz() = delete;

struct quux {
  int quux() = default; // expected-error{{constructor cannot have a return type}}
};

struct attrs {
  [[noreturn]] friend void deleted_with_attrs() = delete;
};
