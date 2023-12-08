// RUN: %clang_cc1 -fsyntax-only -verify %s

struct A {
  void operator delete(void*);
};

namespace NS {
  void operator delete(void *); // expected-error {{'operator delete' cannot be declared inside a namespace}}
}

static void operator delete(void *); // expected-error {{follows non-static declaration}} expected-note {{implicit}}
static void operator delete(void *, int, int); // expected-error {{'operator delete' cannot be declared static in global scope}}
