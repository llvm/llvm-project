#include "warn-static-function-inheader.h"
// RUN: %clang_cc1 -fsyntax-only -verify -Wall %s
// rdar://11202617

static void another(void) { // expected-warning {{function 'another' is not needed and will not be emitted}}
}

template <typename T>
void foo(void) {
  thing();
  another();
}
