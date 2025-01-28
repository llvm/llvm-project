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

#define P_10(x) x, x, x, x, x, x, x, x, x, x,
#define P_100(x) P_10(x) P_10(x) P_10(x) P_10(x) P_10(x) \
                 P_10(x) P_10(x) P_10(x) P_10(x) P_10(x)
#define P_1000(x) P_100(x) P_100(x) P_100(x) P_100(x) P_100(x) \
                  P_100(x) P_100(x) P_100(x) P_100(x) P_100(x)
#define P_10000(x) P_1000(x) P_1000(x) P_1000(x) P_1000(x) P_1000(x) \
                   P_1000(x) P_1000(x) P_1000(x) P_1000(x) P_1000(x)

namespace C1 {
extern "C" int g( 
  P_10000(int = 0) P_10000(int = 0) P_10000(int = 0) P_10000(int = 0) P_10000(int = 0) P_10000(int = 0) P_10000(int = 0) int = 0
  // expected-error@-1 {{too many function parameters; subsequent parameters will be ignored}}
);
} // namespace C1

using C1::g;
int h = g();

void i1(int = 2); // #i1
void i2(int = 2); // #i2
extern "C" void j1(int = 2); // #j1
extern "C" void j2(int = 2); // #j2

void f2() {
  void i1(int = 3); // #i1-redecl
  extern void i2(int = 3); // #i2-redecl
  void j1(int = 3); // #j1-redecl
  extern void j2(int = 3); // #j2-redecl

  i1();
  // expected-error@-1 {{function call relies on ambiguous default argument}}
  //   expected-note@#i1-redecl {{default argument declared here}}
  //   expected-note@#i1 {{default argument declared here}}
  ::i1();
  // expected-error@-1 {{function call relies on ambiguous default argument}}
  //   expected-note@#i1 {{default argument declared here}}
  //   expected-note@#i1-redecl {{default argument declared here}}
  i2();
  // expected-error@-1 {{function call relies on ambiguous default argument}}
  //   expected-note@#i2-redecl {{default argument declared here}}
  //   expected-note@#i2 {{default argument declared here}}
  ::i2();
  // expected-error@-1 {{function call relies on ambiguous default argument}}
  //   expected-note@#i2 {{default argument declared here}}
  //   expected-note@#i2-redecl {{default argument declared here}}
  j1();
  // expected-error@-1 {{function call relies on ambiguous default argument}}
  //   expected-note@#j1-redecl {{default argument declared here}}
  //   expected-note@#j1 {{default argument declared here}}
  ::j1();
  // expected-error@-1 {{function call relies on ambiguous default argument}}
  //   expected-note@#j1 {{default argument declared here}}
  //   expected-note@#j1-redecl {{default argument declared here}}
  j2();
  // expected-error@-1 {{function call relies on ambiguous default argument}}
  //   expected-note@#j2-redecl {{default argument declared here}}
  //   expected-note@#j2 {{default argument declared here}}
  ::j2();
  // expected-error@-1 {{function call relies on ambiguous default argument}}
  //   expected-note@#j2 {{default argument declared here}}
  //   expected-note@#j2-redecl {{default argument declared here}}
}
