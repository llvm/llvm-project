// RUN: %clang_cc1 -std=c++11 -verify %s
//
// Note: [class.inhctor] was removed by P0136R1. This tests the new behavior
// for the wording that used to be there.

template<int> struct X {};

// Constructor characteristics are:
//   - the template parameter list
//   - the parameter-type-list
//   - absence or presence of explicit
//   - absence or presence of constexpr
struct A {
  A(X<0>) {} // expected-note 4{{here}}
  constexpr A(X<1>) {}
  explicit A(X<2>) {} // expected-note 6{{here}}
  explicit constexpr A(X<3>) {} // expected-note 4{{here}}
};

A a0 { X<0>{} };
A a0i = { X<0>{} };
constexpr A a0c { X<0>{} }; // expected-error {{must be initialized by a constant expression}} expected-note {{non-constexpr}}
constexpr A a0ic = { X<0>{} }; // expected-error {{must be initialized by a constant expression}} expected-note {{non-constexpr}}

A a1 { X<1>{} };
A a1i = { X<1>{} };
constexpr A a1c { X<1>{} };
constexpr A a1ic = { X<1>{} };

A a2 { X<2>{} };
A a2i = { X<2>{} }; // expected-error {{constructor is explicit}}
constexpr A a2c { X<2>{} }; // expected-error {{must be initialized by a constant expression}} expected-note {{non-constexpr}}
constexpr A a2ic = { X<2>{} }; // expected-error {{constructor is explicit}}

A a3 { X<3>{} };
A a3i = { X<3>{} }; // expected-error {{constructor is explicit}}
constexpr A a3c { X<3>{} };
constexpr A a3ic = { X<3>{} }; // expected-error {{constructor is explicit}}


struct B : A {
  using A::A;
};

B b0 { X<0>{} };
B b0i = { X<0>{} };
constexpr B b0c { X<0>{} }; // expected-error {{must be initialized by a constant expression}} expected-note {{non-constexpr}}
constexpr B b0ic = { X<0>{} }; // expected-error {{must be initialized by a constant expression}} expected-note {{non-constexpr}}

B b1 { X<1>{} };
B b1i = { X<1>{} };
constexpr B b1c { X<1>{} };
constexpr B b1ic = { X<1>{} };

B b2 { X<2>{} };
B b2i = { X<2>{} }; // expected-error {{constructor is explicit}}
constexpr B b2c { X<2>{} }; // expected-error {{must be initialized by a constant expression}} expected-note {{non-constexpr}}
constexpr B b2ic = { X<2>{} }; // expected-error {{constructor is explicit}}

B b3 { X<3>{} };
B b3i = { X<3>{} }; // expected-error {{constructor is explicit}}
constexpr B b3c { X<3>{} };
constexpr B b3ic = { X<3>{} }; // expected-error {{constructor is explicit}}


// 'constexpr' is OK even if the constructor doesn't obey the constraints.
struct NonLiteral { NonLiteral(); };
struct NonConstexpr { NonConstexpr(); constexpr NonConstexpr(int); };
struct Constexpr { constexpr Constexpr(int) {} };

struct BothNonLiteral : NonLiteral, Constexpr { using Constexpr::Constexpr; }; // expected-note {{base class 'NonLiteral' of non-literal type}}
constexpr BothNonLiteral bothNL{42}; // expected-error {{constexpr variable cannot have non-literal type 'const BothNonLiteral'}}

// FIXME: This diagnostic is not very good. We should explain that the problem is that base class NonConstexpr cannot be initialized.
struct BothNonConstexpr
    : NonConstexpr,
      Constexpr {
  using Constexpr::Constexpr; // expected-note {{here}}
};
constexpr BothNonConstexpr bothNC{42}; // expected-error {{must be initialized by a constant expression}} expected-note {{inherited from base class 'Constexpr'}}


struct ConstexprEval {
  constexpr ConstexprEval(int a, const char *p) : k(p[a]) {}
  char k;
};
struct ConstexprEval2 {
  char k2 = 'x';
};
struct ConstexprEval3 : ConstexprEval, ConstexprEval2 {
  using ConstexprEval::ConstexprEval;
};
constexpr ConstexprEval3 ce{4, "foobar"};
static_assert(ce.k == 'a', "");
static_assert(ce.k2 == 'x', "");


struct TemplateCtors { // expected-note 2{{candidate constructor (the implicit}}
  constexpr TemplateCtors() {}
  template<template<int> class T> TemplateCtors(X<0>, T<0>); // expected-note {{here}} expected-note {{candidate inherited constructor}}
  template<int N> TemplateCtors(X<1>, X<N>); // expected-note {{here}} expected-note {{candidate inherited constructor}}
  template<typename T> TemplateCtors(X<2>, T); // expected-note {{here}} expected-note {{candidate inherited constructor}}

  template<typename T = int> TemplateCtors(int, int = 0, int = 0);
};

struct UsingTemplateCtors : TemplateCtors { // expected-note 3{{candidate constructor (the implicit}}
  using TemplateCtors::TemplateCtors; // expected-note 5{{inherited here}}

  constexpr UsingTemplateCtors(X<0>, X<0>) {} // expected-note {{not viable}}
  constexpr UsingTemplateCtors(X<1>, X<1>) {} // expected-note {{not viable}}
  constexpr UsingTemplateCtors(X<2>, X<2>) {} // expected-note {{not viable}}

  template<int = 0> constexpr UsingTemplateCtors(int) {} // expected-note {{not viable}}
  template<typename T = void> constexpr UsingTemplateCtors(int, int) {} // expected-note {{not viable}}
  template<typename T, typename U> constexpr UsingTemplateCtors(int, int, int) {} // expected-note {{couldn't infer}}
};

template<int> struct Y {};
constexpr UsingTemplateCtors uct1{ X<0>{}, X<0>{} };
constexpr UsingTemplateCtors uct2{ X<0>{}, Y<0>{} }; // expected-error {{must be initialized by a constant expression}} expected-note {{non-constexpr}}
constexpr UsingTemplateCtors uct3{ X<1>{}, X<0>{} }; // expected-error {{must be initialized by a constant expression}} expected-note {{non-constexpr}}
constexpr UsingTemplateCtors uct4{ X<1>{}, X<1>{} };
constexpr UsingTemplateCtors uct5{ X<2>{}, 0 }; // expected-error {{must be initialized by a constant expression}} expected-note {{non-constexpr}}
constexpr UsingTemplateCtors uct6{ X<2>{}, X<2>{} };

constexpr UsingTemplateCtors utc7{ 0 }; // ok
constexpr UsingTemplateCtors utc8{ 0, 0 }; // ok
// FIXME: The standard says that UsingTemplateCtors' (int, int, int) constructor
// hides the one from TemplateCtors, even though the template parameter lists
// don't match. It's not clear that that's *really* the intent, and it's not
// what other compilers do.
constexpr UsingTemplateCtors utc9{ 0, 0, 0 }; // expected-error {{no matching constructor}}
