// RUN: %clang_cc1 -x objective-c %s -fsyntax-only -verify

#include <stdarg.h>

void f1(id arg) {
  NSLog(@"%@", arg); // expected-error {{call to undeclared library function 'NSLog' with type 'void (id, ...)'}} \
  // expected-note {{include the header <Foundation/NSObjCRuntime.h> or explicitly provide a declaration for 'NSLog'}}
}

void f2(id str, va_list args) {
  NSLogv(@"%@", args); // expected-error {{call to undeclared library function 'NSLogv' with type }} \
  // expected-note {{include the header <Foundation/NSObjCRuntime.h> or explicitly provide a declaration for 'NSLogv'}}
}
