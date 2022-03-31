// RUN: %clang_cc1 -fsyntax-only -verify %s
@interface A
@end

void f(void) {
  (A){ 0 }; // expected-error{{cannot initialize Objective-C class type 'A'}}
}
