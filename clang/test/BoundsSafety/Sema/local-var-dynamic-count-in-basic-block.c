
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
#include <ptrcheck.h>
void Test() {
  int len;

  if (len) {
    int *__counted_by(len) ptr; // expected-error{{argument of '__counted_by' attribute cannot refer to declaration from a different scope}}
  }
  // expected-error@+2 3{{local variable len2 must be declared right next to its dependent decl}}
  // expected-error@+3 3{{local variable ptr2 must be declared right next to its dependent decl}} rdar://115657607
  int len2;
  // expected-note@+1 4{{previous use is here}}
  int *__counted_by(len2) ptr2;
  int *__counted_by(len2) ptr3; // expected-error{{variable 'len2' referred to by __counted_by variable cannot be used in other dynamic bounds attributes}}
  int *__counted_by_or_null(len2) ptr4; // expected-error{{variable 'len2' referred to by __counted_by_or_null variable cannot be used in other dynamic bounds attributes}}
  int *__sized_by(len2) ptr5; // expected-error{{variable 'len2' referred to by __sized_by variable cannot be used in other dynamic bounds attributes}}
  int *__sized_by_or_null(len2) ptr6; // expected-error{{variable 'len2' referred to by __sized_by_or_null variable cannot be used in other dynamic bounds attributes}}
}
