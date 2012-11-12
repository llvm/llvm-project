// RUN: %clang_cc1 -fobjc-arc -isystem %S/Inputs %s -verify

#include <arc-system-header.h>

void f(A* a) {
  a->data.void_ptr = 0;
  a->data.a_b.b = 0; // expected-error{{'a_b' is unavailable: this system field has retaining ownership}}
}
// Silly location below
// expected-note{{declaration has been explicitly marked unavailable here}}
