// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

@class NSString;
id a;
NSString *b;

void f() {
  bool b1 = a;
  bool b2 = b;
}


