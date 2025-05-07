

// RUN: %clang_cc1 -fbounds-safety -verify %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -x c -verify %s
// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -x c++ -verify %s
// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -x objective-c -verify %s
// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -x objective-c++ -verify %s

#include <ptrcheck.h>

void foo(int *__counted_by(len) ptr, int len) {
  __auto_type p = ptr; // expected-error{{passing '__counted_by' pointer as __auto_type initializer is not yet supported}}
}
