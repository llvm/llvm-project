

// RUN: %clang_cc1 -O0 -triple arm64-apple-iphoneos -fbounds-safety -emit-llvm %s -o -
// RUN: %clang_cc1 -O0 -triple arm64-apple-iphoneos -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -emit-llvm %s -o -

#include <ptrcheck.h>

enum Enum {ZERO=0, ONE};

void foo(int *__bidi_indexable ptr, enum Enum e) {
  *(ptr + e);
}
