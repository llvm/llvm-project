
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

struct Empty {
  int a[__null_terminated 0]; // expected-error{{'__terminated_by' attribute cannot be applied to empty arrays}}
};

void empty(void) {
  int a[__null_terminated 0];    // expected-error{{'__terminated_by' attribute cannot be applied to empty arrays}}
  int b[__null_terminated] = {}; // expected-error{{incomplete array 'b' with '__terminated_by' attribute must be initialized with at least one element}}
}
