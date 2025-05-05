// RUN: %clang_cc1 -fsyntax-only -verify -Wc++-hidden-decl %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wc++-compat %s
// RUN: %clang_cc1 -fsyntax-only -verify=good %s
// RUN: %clang_cc1 -fsyntax-only -verify=cxx -x c++ -std=c++2c %s
// good-no-diagnostics

struct A {
  struct B { // #b-decl
    int x;
  } bs;
  enum E { // #e-decl
    One
  } es;
  int y;
};

struct C {
  struct D {
    struct F { // #f-decl
	  int x;
    } f;
  } d;
};

struct B b; // expected-warning {{struct defined within a struct or union is not visible in C++}} \
               expected-note@#b-decl {{declared here}} \
               cxx-error {{variable has incomplete type 'struct B'}} \
               cxx-note 3 {{forward declaration of 'B'}}
enum E e;   // expected-warning {{enum defined within a struct or union is not visible in C++}} \
               expected-note@#e-decl {{declared here}} \
               cxx-error {{ISO C++ forbids forward references to 'enum' types}} \
               cxx-error {{variable has incomplete type 'enum E'}} \
               cxx-note 3 {{forward declaration of 'E'}}
struct F f; // expected-warning {{struct defined within a struct or union is not visible in C++}} \
               expected-note@#f-decl {{declared here}} \
               cxx-error {{variable has incomplete type 'struct F'}} \
               cxx-note {{forward declaration of 'F'}}

void func(struct B b);      // expected-warning {{struct defined within a struct or union is not visible in C++}} \
                               expected-note@#b-decl {{declared here}}
void other_func(enum E e) { // expected-warning {{enum defined within a struct or union is not visible in C++}} \
                               expected-note@#e-decl {{declared here}} \
                               cxx-error {{variable has incomplete type 'enum E'}}
  struct B b;               // expected-warning {{struct defined within a struct or union is not visible in C++}} \
                               expected-note@#b-decl {{declared here}} \
                               cxx-error {{variable has incomplete type 'struct B'}}
}

struct X {
  struct B b; // expected-warning {{struct defined within a struct or union is not visible in C++}} \
                 expected-note@#b-decl {{declared here}} \
                 cxx-error {{field has incomplete type 'struct B'}}
  enum E e;   // expected-warning {{enum defined within a struct or union is not visible in C++}} \
                 expected-note@#e-decl {{declared here}} \
                 cxx-error {{field has incomplete type 'enum E'}}
};

struct Y {
  struct Z1 {
    int x;
  } zs;
  
  struct Z2 {
	// This is fine, it is still valid C++.
    struct Z1 inner_zs;
  } more_zs;
};

