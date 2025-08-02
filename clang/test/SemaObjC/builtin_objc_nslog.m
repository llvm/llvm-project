// RUN: %clang_cc1 -x objective-c %s -fsyntax-only -verify

#include <stdarg.h>

void f1(id arg) {
  NSLog(@"%@", arg); // expected-error {{call to undeclared library function 'NSLog', will assume it exists with type 'void (id, ...)'; ISO C99 and later do not support implicit function declaration}} \
  // expected-note {{include the header <Foundation/NSObjCRuntime.h> or explicitly provide a declaration for 'NSLog'}}
}

void f2(id str, va_list args) {
  NSLogv(@"%@", args); // expected-error {{call to undeclared library function 'NSLogv', will assume it exists with type 'void (id, __builtin_va_list)'; ISO C99 and later do not support implicit function declarations}} \
  // expected-note {{include the header <Foundation/NSObjCRuntime.h> or explicitly provide a declaration for 'NSLogv'}}
}
