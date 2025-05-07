// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

// expected-note@+1{{consider adding '__sized_by(10)' to 'dst'}}
void foo(char *dst, char *src) {
    // expected-error@+1{{passing 'char *__single' with pointee of size 1 to parameter of type 'void *__single __sized_by(function-parameter-0-2)' (aka 'void *__single') with size value of 10 always fails}}
    __builtin_memcpy(dst, src, 10);
}
