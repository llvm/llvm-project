// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++2c %s

namespace A {
  extern "C" void f1(...);
  extern "C" void f2(int, ...);
  extern "C" void f3(int = 0, ...); // #A-f3
} // namespace A

namespace B {
  extern "C" void f1(...);
  extern "C" void f2(int, ...); // #B-f2
  extern "C" void f3(int = 0, ...); // #B-f3
} // namespace B

void f() {
  using A::f1;
  using A::f2;
  using A::f3;
  using B::f1;
  using B::f2;
  using B::f3;

  f1();
  f1(0);
  f1(0, 0);
  f2();
  // expected-error@-1 {{no matching function for call to 'f2'}}
  //   expected-note@#B-f2 {{candidate function not viable: requires at least 1 argument, but 0 were provided}}
  f2(0);
  f2(0, 0);
  f3();
  // expected-error@-1 {{function call relies on ambiguous default argument}}
  //   expected-note@#B-f3 {{default argument declared here}}
  //   expected-note@#A-f3 {{default argument declared here}}
  f3(0);
  f3(0, 0);
}
