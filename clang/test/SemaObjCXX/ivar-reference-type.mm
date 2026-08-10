// RUN: %clang_cc1 -fsyntax-only -verify %s
@interface A {
  int &r; // expected-error {{instance variables cannot be of reference type}}
}
@end
