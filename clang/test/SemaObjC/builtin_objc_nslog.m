// RUN: %clang_cc1 -x objective-c %s -fsyntax-only -verify

#include <stdarg.h>

void f1(id arg) {
  NSLog(@"%@", arg); // expected-warning {{implicitly declaring library function 'NSLog' with type 'void (id, ...)'}} \
  // expected-note {{please include the header <Foundation/NSObjCRuntime.h> or explicitly provide a declaration for 'NSLog'}}
}

void f2(id str, va_list args) {
  NSLogv(@"%@", args); // expected-warning {{implicitly declaring library function 'NSLogv' with type }} \
  // expected-note {{please include the header <Foundation/NSObjCRuntime.h> or explicitly provide a declaration for 'NSLogv'}}
}
