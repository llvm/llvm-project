
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

int foo(void) {
  int array[3] = {1, 2, 3};
  int(*ap)[3] = &array;
  return (*ap)[0];
}

int bar(void) {
  int array[3] = {1, 2, 3};
  int *p = array;
  // expected-warning@+1{{incompatible pointer types initializing 'int (*__bidi_indexable)[3]' with an expression of type 'int *__bidi_indexable'}}
  int(*ap)[3] = p;
  return (*ap)[0];
}
