// Based on C++20 10.2 example 5.

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/std-10-2-ex5-tu1.cpp \
// RUN:  -o  %t/M.pcm

// RUN: %clang_cc1 -std=c++20 -emit-obj %t/std-10-2-ex5-tu2.cpp \
// RUN:  -fmodule-file=M=%t/M.pcm -o  %t/tu-2.o

// RUN: %clang_cc1 -std=c++20 -emit-obj %t/std-10-2-ex5-tu3.cpp \
// RUN:  -fmodule-file=M=%t/M.pcm -verify -o %t/main.o

//--- std-10-2-ex5-tu1.cpp
export module M;
export struct X {
  static void f();
  struct Y {};
};
namespace {
struct S {};
} // namespace
export void f(S); // OK
struct T {};
export T id(T);  // OK
export struct A; // A exported as incomplete

export auto rootFinder(double a) {
  return [=](double x) { return (x + a / x) / 2; };
}
export const int n = 5; // OK, n has external linkage

//--- std-10-2-ex5-tu2.cpp

module M;
struct A {
  int value;
};

//--- std-10-2-ex5-tu3.cpp

import M;

int main() {
  X::f();                 // OK, X is exported and definition of X is reachable
  X::Y y;                 // OK, X::Y is exported as a complete type
  auto f = rootFinder(2); // OK
                          // error: A is incomplete
  return A{45}.value;     // expected-error {{invalid use of incomplete type 'A'}}
                          // expected-error@-1 {{member access into incomplete type 'A'}}
                          // expected-note@std-10-2-ex5-tu1.cpp:12 2{{forward declaration of 'A'}}
}
