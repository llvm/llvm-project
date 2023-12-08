// RUN: %clang_cc1 -fsyntax-only -pedantic -std=c++11 -verify %s
// expected-no-diagnostics

void foo();

void bar() { };

void wibble();

;

namespace Blah {
  void f() { };
  
  void g();
}
