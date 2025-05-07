
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

int array[10] = {0};

const int * __bidi_indexable const bidx = &array[9] - 10;

const int * __indexable const idxa = bidx;
// expected-error@-1{{initializer element is not a compile-time constant}}
