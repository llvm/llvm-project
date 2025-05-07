
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

void none_none(void) {
  int len1;
  int *__counted_by(len1) ptr1;

  int len2;
  int *__sized_by(len2) ptr2;
}

void none_static(void) {
  int len1;
  static int *__counted_by(len1) ptr1; // expected-error{{argument of '__counted_by' attribute cannot refer to declaration of a different lifetime}}

  int len2;
  static int *__sized_by(len2) ptr2; // expected-error{{argument of '__sized_by' attribute cannot refer to declaration of a different lifetime}}
}

void none_auto(void) {
  int len1;
  auto int *__counted_by(len1) ptr1;

  int len2;
  auto int *__sized_by(len2) ptr2;
}

void none_register(void) {
  int len1;
  register int *__counted_by(len1) ptr1;

  int len2;
  register int *__sized_by(len2) ptr2;
}

void static_none(void) {
  static int len1;
  int *__counted_by(len1) ptr1; // expected-error{{argument of '__counted_by' attribute cannot refer to declaration of a different lifetime}}

  static int len2;
  int *__sized_by(len2) ptr2; // expected-error{{argument of '__sized_by' attribute cannot refer to declaration of a different lifetime}}
}

void static_static(void) {
  static int len1;
  static int *__counted_by(len1) ptr1;

  static int len2;
  static int *__sized_by(len2) ptr2;
}

void static_auto(void) {
  static int len1;
  auto int *__counted_by(len1) ptr1; // expected-error{{argument of '__counted_by' attribute cannot refer to declaration of a different lifetime}}

  static int len2;
  auto int *__sized_by(len2) ptr2; // expected-error{{argument of '__sized_by' attribute cannot refer to declaration of a different lifetime}}
}

void static_register(void) {
  static int len1;
  register int *__counted_by(len1) ptr1; // expected-error{{argument of '__counted_by' attribute cannot refer to declaration of a different lifetime}}

  static int len2;
  register int *__sized_by(len2) ptr2; // expected-error{{argument of '__sized_by' attribute cannot refer to declaration of a different lifetime}}
}

void auto_none(void) {
  auto int len1;
  int *__counted_by(len1) ptr1;

  auto int len2;
  int *__sized_by(len2) ptr2;
}

void auto_static(void) {
  auto int len1;
  static int *__counted_by(len1) ptr1; // expected-error{{argument of '__counted_by' attribute cannot refer to declaration of a different lifetime}}

  auto int len2;
  static int *__sized_by(len2) ptr2; // expected-error{{argument of '__sized_by' attribute cannot refer to declaration of a different lifetime}}
}

void auto_auto(void) {
  auto int len1;
  auto int *__counted_by(len1) ptr1;

  auto int len2;
  auto int *__sized_by(len2) ptr2;
}

void auto_register(void) {
  auto int len1;
  register int *__counted_by(len1) ptr1;

  auto int len2;
  register int *__sized_by(len2) ptr2;
}

void register_none(void) {
  register int len1;
  int *__counted_by(len1) ptr1;

  register int len2;
  int *__sized_by(len2) ptr2;
}

void register_static(void) {
  register int len1;
  static int *__counted_by(len1) ptr1; // expected-error{{argument of '__counted_by' attribute cannot refer to declaration of a different lifetime}}

  register int len2;
  static int *__sized_by(len2) ptr2; // expected-error{{argument of '__sized_by' attribute cannot refer to declaration of a different lifetime}}
}

void register_auto(void) {
  register int len1;
  auto int *__counted_by(len1) ptr1;

  register int len2;
  auto int *__sized_by(len2) ptr2;
}

void register_register(void) {
  register int len1;
  register int *__counted_by(len1) ptr1;

  register int len2;
  register int *__sized_by(len2) ptr2;
}

void extern_none(void) {
  extern int len1;
  int *__counted_by(len1) ptr1; // expected-error{{argument of '__counted_by' attribute cannot refer to declaration of a different lifetime}}

  extern int len2;
  int *__sized_by(len2) ptr2; // expected-error{{argument of '__sized_by' attribute cannot refer to declaration of a different lifetime}}
}

void extern_static(void) {
  extern int len1;
  static int *__counted_by(len1) ptr1; // expected-error{{argument of '__counted_by' attribute cannot refer to declaration of a different lifetime}}

  extern int len2;
  static int *__sized_by(len2) ptr2; // expected-error{{argument of '__sized_by' attribute cannot refer to declaration of a different lifetime}}
}

void extern_auto(void) {
  extern int len1;
  auto int *__counted_by(len1) ptr1; // expected-error{{argument of '__counted_by' attribute cannot refer to declaration of a different lifetime}}

  extern int len2;
  auto int *__sized_by(len2) ptr2; // expected-error{{argument of '__sized_by' attribute cannot refer to declaration of a different lifetime}}
}

void extern_register(void) {
  extern int len1;
  register int *__counted_by(len1) ptr1; // expected-error{{argument of '__counted_by' attribute cannot refer to declaration of a different lifetime}}

  extern int len2;
  register int *__sized_by(len2) ptr2; // expected-error{{argument of '__sized_by' attribute cannot refer to declaration of a different lifetime}}
}

void private_extern_none(void) {
  // expected-warning@+2{{use of __private_extern__ on a declaration may not produce external symbol private to the linkage unit and is deprecated}}
  // expected-note@+1{{use __attribute__((visibility("hidden"))) attribute instead}}
  __private_extern__ int len1;
  int *__counted_by(len1) ptr1; // expected-error{{argument of '__counted_by' attribute cannot refer to declaration of a different lifetime}}

  // expected-warning@+2{{use of __private_extern__ on a declaration may not produce external symbol private to the linkage unit and is deprecated}}
  // expected-note@+1{{use __attribute__((visibility("hidden"))) attribute instead}}
  __private_extern__ int len2;
  int *__sized_by(len2) ptr2; // expected-error{{argument of '__sized_by' attribute cannot refer to declaration of a different lifetime}}
}

void private_extern_static(void) {
  // expected-warning@+2{{use of __private_extern__ on a declaration may not produce external symbol private to the linkage unit and is deprecated}}
  // expected-note@+1{{use __attribute__((visibility("hidden"))) attribute instead}}
  __private_extern__ int len1;
  static int *__counted_by(len1) ptr1; // expected-error{{argument of '__counted_by' attribute cannot refer to declaration of a different lifetime}}

  // expected-warning@+2{{use of __private_extern__ on a declaration may not produce external symbol private to the linkage unit and is deprecated}}
  // expected-note@+1{{use __attribute__((visibility("hidden"))) attribute instead}}
  __private_extern__ int len2;
  static int *__sized_by(len2) ptr2; // expected-error{{argument of '__sized_by' attribute cannot refer to declaration of a different lifetime}}
}

void private_extern_auto(void) {
  // expected-warning@+2{{use of __private_extern__ on a declaration may not produce external symbol private to the linkage unit and is deprecated}}
  // expected-note@+1{{use __attribute__((visibility("hidden"))) attribute instead}}
  __private_extern__ int len1;
  auto int *__counted_by(len1) ptr1; // expected-error{{argument of '__counted_by' attribute cannot refer to declaration of a different lifetime}}

  // expected-warning@+2{{use of __private_extern__ on a declaration may not produce external symbol private to the linkage unit and is deprecated}}
  // expected-note@+1{{use __attribute__((visibility("hidden"))) attribute instead}}
  __private_extern__ int len2;
  auto int *__sized_by(len2) ptr2; // expected-error{{argument of '__sized_by' attribute cannot refer to declaration of a different lifetime}}
}

void private_extern_register(void) {
  // expected-warning@+2{{use of __private_extern__ on a declaration may not produce external symbol private to the linkage unit and is deprecated}}
  // expected-note@+1{{use __attribute__((visibility("hidden"))) attribute instead}}
  __private_extern__ int len1;
  register int *__counted_by(len1) ptr1; // expected-error{{argument of '__counted_by' attribute cannot refer to declaration of a different lifetime}}

  // expected-warning@+2{{use of __private_extern__ on a declaration may not produce external symbol private to the linkage unit and is deprecated}}
  // expected-note@+1{{use __attribute__((visibility("hidden"))) attribute instead}}
  __private_extern__ int len2;
  register int *__sized_by(len2) ptr2; // expected-error{{argument of '__sized_by' attribute cannot refer to declaration of a different lifetime}}
}
