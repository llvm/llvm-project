

// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

// expected-no-diagnostics

void foo(int len, int *__counted_by(len) p);

__typeof__(foo) foo;
extern __attribute__((weak_import)) __typeof__(foo) foo;

void bar(int len, int *__counted_by(len) p) { foo(len, p); }
