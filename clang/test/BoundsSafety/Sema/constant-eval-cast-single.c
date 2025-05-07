// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x objective-c -verify %s

#include <ptrcheck.h>
#include <stdint.h>

int32_t *__single p = (int32_t *__single)(int64_t *__single)0;

// expected-error@+1{{initializer element is not a compile-time constant}}
int32_t *__single q = (int32_t *__single)(int16_t *__single)0;
