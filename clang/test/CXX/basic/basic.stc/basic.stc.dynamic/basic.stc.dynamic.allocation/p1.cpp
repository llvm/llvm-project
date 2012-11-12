// RUN: %clang_cc1 -fsyntax-only -verify %s
#include <stddef.h>

struct A {
  void *operator new(size_t);
};

namespace NS {
  void *operator new(size_t);; // expected-error {{'operator new' cannot be declared inside a namespace}}
}

static void *operator new(size_t); // expected-error {{'operator new' cannot be declared static in global scope}}

struct B {
  void operator new(size_t);  // expected-error {{'operator new' must return type 'void *'}}
};

struct C {
  void *operator new(); // expected-error {{'operator new' must have at least one parameter}}
};

struct D {
  void *operator new(bool); // expected-error {{'operator new' takes type size_t}}
};

struct E {
  void *operator new(size_t = 0); // expected-error {{parameter of 'operator new' cannot have a default argument}}
};

struct F {
  template<typename T> void *operator new(size_t, int);
};

struct G {
  template<typename T> T operator new(size_t, int); // expected-error {{'operator new' cannot have a dependent return type; use 'void *' instead}}
};

struct H {
  template<typename T> void *operator new(T, int); // expected-error {{'operator new' cannot take a dependent type as first parameter; use size_t}}
};

struct I {
  template<typename T> void *operator new(size_t); // expected-error {{'operator new' template must have at least two parameters}}
};
