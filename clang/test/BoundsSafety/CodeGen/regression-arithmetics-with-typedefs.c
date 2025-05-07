

// RUN: %clang_cc1 -O0 -triple arm64-apple-iphoneos -fbounds-safety -emit-llvm %s -o -
// RUN: %clang_cc1 -O0 -triple arm64-apple-iphoneos -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -emit-llvm %s -o -

#include <ptrcheck.h>

typedef unsigned long size_t;

void foo(int *__bidi_indexable ptr, size_t idx) {
  *(ptr + idx);
}
