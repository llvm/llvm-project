
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

int glen;
extern int extglen;

void ptr(void) {
  extern int len1;
  extern int *__counted_by(len1) ptr1;

  extern int len2;
  extern int *__sized_by(len2) ptr2;

  int len3;
  // expected-error@+3{{argument of '__counted_by' attribute cannot refer to declaration of a different lifetime}}
  // expected-warning@+2{{use of __private_extern__ on a declaration may not produce external symbol private to the linkage unit and is deprecated}}
  // expected-note@+1{{use __attribute__((visibility("hidden"))) attribute instead}}
  __private_extern__ int *__counted_by(len3) ptr3;

  extern int len4;
  // expected-error@+3{{argument of '__sized_by' attribute cannot refer to declaration of a different lifetime}}
  // expected-warning@+2{{use of __private_extern__ on a declaration may not produce external symbol private to the linkage unit and is deprecated}}
  // expected-note@+1{{use __attribute__((visibility("hidden"))) attribute instead}}
  __private_extern__ int *__sized_by(len4) ptr4;

  // expected-error@+4{{argument of '__sized_by' attribute cannot refer to declaration of a different lifetime}}
  // expected-error@+3{{argument of '__sized_by' attribute cannot refer to declaration from a different scope}}
  // expected-warning@+2{{use of __private_extern__ on a declaration may not produce external symbol private to the linkage unit and is deprecated}}
  // expected-note@+1{{use __attribute__((visibility("hidden"))) attribute instead}}
  __private_extern__ int *__sized_by(extglen) ptr5;

}

void fptr(void) {
  extern int *__counted_by(42) (*fptr1)(void);

  extern int *__sized_by(42) (*fptr2)(void);

  // expected-warning@+2{{use of __private_extern__ on a declaration may not produce external symbol private to the linkage unit and is deprecated}}
  // expected-note@+1{{use __attribute__((visibility("hidden"))) attribute instead}}
  __private_extern__ int *__counted_by(42) (*fptr3)(void);

  // expected-warning@+2{{use of __private_extern__ on a declaration may not produce external symbol private to the linkage unit and is deprecated}}
  // expected-note@+1{{use __attribute__((visibility("hidden"))) attribute instead}}
  __private_extern__ int *__sized_by(42) (*fptr4)(void);
}

void incomplete_array(void) {
  extern int array1[__counted_by(42)];

  // expected-warning@+2{{use of __private_extern__ on a declaration may not produce external symbol private to the linkage unit and is deprecated}}
  // expected-note@+1{{use __attribute__((visibility("hidden"))) attribute instead}}
  __private_extern__ int array2[__counted_by(42)];

  extern int len3;
  // expected-error@+3{{argument of '__counted_by' attribute cannot refer to declaration of a different lifetime}}
  // expected-warning@+2{{use of __private_extern__ on a declaration may not produce external symbol private to the linkage unit and is deprecated}}
  // expected-note@+1{{use __attribute__((visibility("hidden"))) attribute instead}}
  __private_extern__ int array3[__counted_by(len3)];

}
