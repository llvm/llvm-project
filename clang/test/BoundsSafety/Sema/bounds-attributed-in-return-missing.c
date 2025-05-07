

// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c++ -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x objective-c -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x objective-c++ -verify %s

#include <ptrcheck.h>

// The warnings should be always errors (rdar://136173954).

// expected-warning@+1{{non-void function does not return a value}}
int *__counted_by(42) missing(void) {}

// expected-warning@+1{{non-void function does not return a value}}
int *__counted_by(count) missing2(int count) {}

int *__counted_by(42) mismatch(void) {
  return; // expected-error{{non-void function 'mismatch' should return a value}}
}

int *__counted_by(count) mismatch2(int count) {
  return; // expected-error{{non-void function 'mismatch2' should return a value}}
}
