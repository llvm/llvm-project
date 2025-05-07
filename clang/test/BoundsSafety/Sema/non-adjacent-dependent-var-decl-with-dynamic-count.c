
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
#include <ptrcheck.h>
void side_effect();

void Test() {
  int len; // // expected-error{{local variable len must be declared right next to its dependent decl}}
  side_effect();
  int *__counted_by(len) ptr; // expected-error{{local variable ptr must be declared right next to its dependent decl}}
}
