// RUN: %clang_cc1 -std=gnu++98 -fobjc-arc -isystem %S/Inputs %s -verify

#include <arc-system-header.h>

void f(A* a) {
  a->data.void_ptr = 0;
  a->data.a_b.b = 0; // expected-error{{'a_b' is unavailable in ARC}}
}
// expected-note@arc-system-header.h:10{{field has non-trivial ownership qualification}}
