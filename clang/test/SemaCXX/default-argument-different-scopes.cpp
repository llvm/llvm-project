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

void f3(char);
void f4(int = 0); // #f4-global
void f5(int = 0); // #f5-global

void f() {
  using A::f1;
  using B::f1;

  f1();
  f1(0);
  f1(0, 0);

  using A::f2;
  using B::f2;
  
  f2();
  // expected-error@-1 {{no matching function for call to 'f2'}}
  //   expected-note@#B-f2 {{candidate function not viable: requires at least 1 argument, but 0 were provided}}
  f2(0);
  f2(0, 0);

  using A::f3;
  using B::f3;

  f3();
  // expected-error@-1 {{function call relies on default argument that has multiple definitions}}
  //   expected-note@#A-f3 {{default argument declared here}}
  //   expected-note@#B-f3 {{default argument declared here}}
  f3(0);
  f3(0, 0);
  f3('a');

  using ::f4;
  void f4(int = 1); // #f4-local
  f4();
  // expected-error@-1 {{function call relies on default argument that has multiple definitions}}
  //   expected-note@#f4-global {{default argument declared here}}
  //   expected-note@#f4-local {{default argument declared here}}

  using ::f5;
  extern void f5(int = 1); // #f5-local
  extern void f5(int);
  f5();
  // expected-error@-1 {{function call relies on default argument that has multiple definitions}}
  //   expected-note@#f5-global {{default argument declared here}}
  //   expected-note@#f5-local {{default argument declared here}}
}

// Two declarations in different scopes have to be found and be viable
// candidates to run into ambiguous default arguments situation.
// In the tests below, calls with qualified names finds declarations only
// at namespace scope, whereas calls with unqualified names find only
// declarations at block scope. As a result, all calls are well-formed
// in 'i' group of tests.

void i1(int = 2);
void i2(int = 2);
extern "C" void j1(int = 2);
extern "C" void j2(int = 2);

void i() {
  void i1(int = 3);
  ::i1();
  i1();

  extern void i2(int = 3);
  ::i2();
  i2();

  void j1(int = 3);
  ::j1();
  j1();

  extern void j2(int = 3);
  ::j2();
  j2();
}

// In 'k' group of tests, no redefinition of default arguments occur,
// because sets of default arguments are associated with lexical scopes
// of function declarations. There are exceptions from this rule,
// described below.

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

// The only exception from the rule that default arguments are associated with
// their lexical scope is out-of-line definitions of member functions of
// non-templated classes. Such default arguments contribute to the set of
// default arguments associated with the class scope.

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

// Default arguments are not allowed in out-of-line declarations
// of member functions of class templates. They have to be specified within
// member-specification.

template <typename>
struct M {
  void m1(int);
  void m2(int = 2);
  void m3(int = 3); // #m3
};

template <typename T>
void M<T>::m1(int = 1) {}
// expected-error@-1 {{default arguments cannot be added to an out-of-line definition of a member of a class template}}

template <typename T>
void M<T>::m2(int) {}

// FIXME: the real problem is that default argument is not allowed here,
//        and not that it's redefined.
template <typename T>
void M<T>::m3(int = 3) {}
// expected-error@-1 {{redefinition of default argument}}
//   expected-note@#m3 {{previous definition is here}}
