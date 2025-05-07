
// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-intrinsics -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-intrinsics -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

struct Foo {
  char *__bidi_indexable __ptrauth(1, 1, 1) inner1;
  // expected-error@-1 {{pointer authentication is currently unsupported on indexable pointers}}
  char *__indexable __ptrauth(1, 1, 1) inner2;
  // expected-error@-1 {{pointer authentication is currently unsupported on indexable pointers}}
  char *__ptrauth(1, 1, 1) __bidi_indexable inner3;
  // expected-error@-1 {{pointer authentication is currently unsupported on indexable pointers}}
  char *__ptrauth(1, 1, 1) __indexable inner4;
  // expected-error@-1 {{pointer authentication is currently unsupported on indexable pointers}}
  char *__ptrauth(1, 1, 1) __single inner5;
  char *__ptrauth(1, 1, 1) __unsafe_indexable inner6;
};
