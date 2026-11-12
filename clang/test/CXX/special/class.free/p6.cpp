// RUN: %clang_cc1 -fsyntax-only -verify %s
#include <stddef.h>

struct A {
  void operator delete(void*) {
    (void)this; // expected-error {{invalid use of 'this' outside of a non-static member function}}
  }
  void operator delete[](void*) {
    (void)this; // expected-error {{invalid use of 'this' outside of a non-static member function}}
  }
};
