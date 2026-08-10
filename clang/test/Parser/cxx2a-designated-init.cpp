// RUN: %clang_cc1 -std=c++98 -verify=cxx98 %s
// RUN: %clang_cc1 -std=c++11 -verify %s -Wno-c++2a-extensions
// RUN: %clang_cc1 -std=c++2a -verify %s

struct A {
  explicit A(int, int); // expected-note {{here}}
};

struct B {
  A a;
};

B b1 = {.a = {1, 2}}; // cxx98-error {{non-aggregate type 'A' cannot be initialized with an initializer list}}
// expected-error@-1 {{chosen constructor is explicit in copy-initialization}}
B b2 = {.a{1, 2}}; // cxx98-error {{expected '='}}

struct C {
  char x, y;
};
struct D {
  C c;
};

D d1 = {.c = {1, 2000}}; // cxx98-warning {{changes value}} expected-error {{narrow}} expected-warning {{changes value}} expected-note {{}}
D d2 = {.c{1, 2000}}; // cxx98-error {{expected '='}} expected-error {{narrow}} expected-warning {{changes value}} expected-note {{}}
