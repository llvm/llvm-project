

// RUN: %clang_cc1 -analyze -fbounds-safety -w -verify -analyzer-checker=core %s
// RUN: %clang_cc1 -analyze -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -w -verify -analyzer-checker=core %s

// expected-no-diagnostics

#include <ptrcheck.h>

void test_void_pointer(void *__bidi_indexable p) {
  void *q = p;
  if (q) {}
  q != 0; // no-crash
}

void test_opaque_pointer(struct S *__bidi_indexable p) {
  struct S *q = p;
  if (!q) {
    q != 0; // no-crash
  }
}
