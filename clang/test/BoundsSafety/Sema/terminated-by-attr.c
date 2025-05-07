
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

// Check if the terminator is an ICE.
void term(int val) {
  char a1[__terminated_by(0)] = "";
  char a2[__terminated_by(0x40 + 1)] = {'A'};
  char a3[__terminated_by(val)] = ""; // expected-error{{'__terminated_by' attribute requires an integer constant}}
}

void multiple_attrs(void) {
  char a1[__null_terminated __null_terminated] = "";    // expected-warning{{array annotated with __terminated_by multiple times. Annotate only once to remove this warning}}
  char a2[__null_terminated __terminated_by(42)] = ""; // expected-error{{array cannot have more than one terminator attribute}}
                                                       // expected-note@-1{{conflicting arguments for terminator were '0' and '42'}}
  char a3[__terminated_by(42)] = ""; // expected-error{{array 'a3' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '0')}}

  char *__null_terminated __null_terminated p1 = "";    // expected-warning{{pointer annotated with __terminated_by multiple times. Annotate only once to remove this warning}}
  char *__null_terminated __terminated_by(42) p2 = ""; // expected-error{{pointer cannot have more than one terminator attribute}}
                                                       // expected-note@-1{{conflicting arguments for terminator were '0' and '42'}}
  char * __terminated_by(0) __terminated_by(1) __terminated_by(2) __terminated_by(0) p3 = "";
                                                       // expected-error@-1 2{{pointer cannot have more than one terminator attribute}}
                                                       // expected-note@-2{{conflicting arguments for terminator were '0' and '1'}}
                                                       // expected-note@-3{{conflicting arguments for terminator were '0' and '2'}}
                                                       // expected-warning@-4{{pointer annotated with __terminated_by multiple times. Annotate only once to remove this warning}}

  char *__null_terminated a4[__null_terminated 1] = {0}; // ok (the attributes apply to different types)
}

void type(int v) {
  char __null_terminated c;     // expected-error{{'__terminated_by' attribute can be applied to pointers, constant-length arrays or incomplete arrays}}
  char a1[__null_terminated v]; // expected-error{{'__terminated_by' attribute can be applied to pointers, constant-length arrays or incomplete arrays}}

  float a2[__null_terminated 1]; // expected-error{{element type of array with '__terminated_by' attribute must be an integer or a non-wide pointer}}
  float *__null_terminated p1;   // expected-error{{pointee type of pointer with '__terminated_by' attribute must be an integer or a non-wide pointer}}

  int(*__null_terminated p2)[1]; // expected-error{{pointee type of pointer with '__terminated_by' attribute must be an integer or a non-wide pointer}}

  struct Foo {
    int x;
  };

  struct Foo a3[__null_terminated 1]; // expected-error{{element type of array with '__terminated_by' attribute must be an integer or a non-wide pointer}}
  struct Foo *__null_terminated p3;   // expected-error{{pointee type of pointer with '__terminated_by' attribute must be an integer or a non-wide pointer}}

  int *__single a4[__null_terminated 1] = {}; // ok
  int *__single *__null_terminated p4 = 0;    // ok

  int *__bidi_indexable a5[__null_terminated 1]; // expected-error{{element type of array with '__terminated_by' attribute must be an integer or a non-wide pointer}}
  int *__bidi_indexable *__null_terminated p5;   // expected-error{{pointee type of pointer with '__terminated_by' attribute must be an integer or a non-wide pointer}}
}

void ptr_attrs(void) {
  char *__null_terminated __single p1 = ""; // ok
  char *__single __null_terminated p2 = ""; // ok

  char *__null_terminated __indexable p3 = ""; // expected-error{{'__terminated_by' attribute currently can be applied only to '__single' pointers}}
  char *__indexable __null_terminated p4 = ""; // expected-error{{'__terminated_by' attribute currently can be applied only to '__single' pointers}}

  char *__null_terminated __bidi_indexable p5 = ""; // expected-error{{'__terminated_by' attribute currently can be applied only to '__single' pointers}}
  char *__bidi_indexable __null_terminated p6 = ""; // expected-error{{'__terminated_by' attribute currently can be applied only to '__single' pointers}}

  char *__null_terminated __unsafe_indexable p7 = ""; // expected-error{{'__terminated_by' attribute currently can be applied only to '__single' pointers}}
  char *__unsafe_indexable __null_terminated p8 = ""; // expected-error{{'__terminated_by' attribute currently can be applied only to '__single' pointers}}

  char *__null_terminated __counted_by(0) p9 = "";  // expected-error{{'__terminated_by' attribute currently can be applied only to '__single' pointers}}
  char *__counted_by(0) __null_terminated p10 = ""; // expected-error{{'__terminated_by' attribute currently can be applied only to '__single' pointers}}

  char *__null_terminated __sized_by(0) p11 = ""; // expected-error{{'__terminated_by' attribute currently can be applied only to '__single' pointers}}
  char *__sized_by(0) __null_terminated p12 = ""; // expected-error{{'__terminated_by' attribute currently can be applied only to '__single' pointers}}

  char *__sized_by(0) p13 = 0;
  __typeof__(p13) __null_terminated p14 = ""; // expected-error{{'__terminated_by' attribute currently can be applied only to '__single' pointers}}
}
