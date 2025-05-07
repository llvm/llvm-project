
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

// Regression test for rdar://133766202
// expected-no-diagnostics

struct S {
  int field1;
  int field2;
};

void foo() {
  struct S arr[2];
  arr->field1 = 3;
}
