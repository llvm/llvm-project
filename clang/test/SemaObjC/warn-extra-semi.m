// RUN: %clang_cc1 -x objective-c++ %s -std=c++11 -fsyntax-only -verify=expected,wextra -Wextra-semi
// RUN: %clang_cc1 -x objective-c++ %s -std=c++11 -fsyntax-only -verify=expected -pedantic

@interface X
{
  ; // expected-warning {{extra ';' inside instance variable list}}
  int a;
  ; // expected-warning {{extra ';' inside instance variable list}}
  ;; // expected-warning {{extra ';' inside instance variable list}}
}
@end
