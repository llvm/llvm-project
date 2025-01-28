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

// In 'k' group of tests, no redefinition of default arguments occur,
// because sets of default arguments are associated with lexical scopes
// of function declarations.

void k1(int); // #k1
void k2(int = 2);
void k3(int = 3); // #k3

struct K {
  friend void k1(int = 1) {}
  // expected-error@-1 {{friend declaration specifying a default argument must be the only declaration}}
  //   expected-note@#k1 {{previous declaration is here}}
  friend void k2(int) {}
  friend void k3(int = 3) {}
  // expected-error@-1 {{friend declaration specifying a default argument must be the only declaration}}
  //   expected-note@#k3 {{previous declaration is here}}

  friend void k4(int = 4) {} // #k4
  friend void k5(int) {}
  friend void k6(int = 6) {} // #k6
};

void k4(int);
// expected-error@-1 {{friend declaration specifying a default argument must be the only declaration}}
//   expected-note@#k4 {{previous declaration is here}}
void k5(int = 5);
void k6(int = 6);
// expected-error@-1 {{friend declaration specifying a default argument must be the only declaration}}
//   expected-note@#k6 {{previous declaration is here}}

struct L {
  void l1(int);
  void l2(int = 2);
  void l3(int = 3); // #l3

  template <typename>
  void l4(int); // #l4
  template <typename>
  void l5(int = 5);
  template <typename>
  void l6(int = 6); // #l6
};

void L::l1(int = 1) {}
void L::l2(int) {}
void L::l3(int = 3) {}
// expected-error@-1 {{redefinition of default argument}}
//   expected-note@#l3 {{previous definition is here}}

template <typename>
void L::l4(int = 4) {}
// expected-error@-1 {{default arguments cannot be added to a function template that has already been declared}}
//   expected-note@#l4 {{previous template declaration is here}}

template <typename>
void L::l5(int) {}

template <typename>
void L::l6(int = 6) {}
// expected-error@-1 {{redefinition of default argument}}
//   expected-note@#l6 {{previous definition is here}}
