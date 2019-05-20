// RUN: %clang_cc1 -verify -std=c++11 -Wno-anonymous-pack-parens %s
// RUN: cp %s %t
// RUN: not %clang_cc1 -x c++ -std=c++11 -fixit %t
// RUN: %clang_cc1 -Wall -pedantic -x c++ -std=c++11 %t

/* This is a test of the various code modification hints that only
   apply in C++0x. */
struct A {
  explicit operator int(); // expected-note{{conversion to integral type}}
};

void x() {
  switch(A()) { // expected-error{{explicit conversion to}}
  }
}

using ::T = void; // expected-error {{name defined in alias declaration must be an identifier}}
using typename U = void; // expected-error {{name defined in alias declaration must be an identifier}}
using typename ::V = void; // expected-error {{name defined in alias declaration must be an identifier}}

namespace SemiCommaTypo {
  int m {},
  n [[]], // expected-error {{expected ';' at end of declaration}}
  int o;

  struct Base {
    virtual void f2(), f3();
  };
  struct MemberDeclarator : Base {
    int k : 4,
        //[[]] : 1, FIXME: test this once we support attributes here
        : 9, // expected-error {{expected ';' at end of declaration}}
    char c, // expected-error {{expected ';' at end of declaration}}
    typedef void F(), // expected-error {{expected ';' at end of declaration}}
    F f1,
      f2 final,
      f3 override, // expected-error {{expected ';' at end of declaration}}
  };
}

namespace ScopedEnum {
  enum class E { a };

  enum class E b = E::a; // expected-error {{must use 'enum' not 'enum class'}}
  struct S {
    friend enum class E; // expected-error {{must use 'enum' not 'enum class'}}
  };
}

struct S2 { 
  void f(int i); 
  void g(int i);
};

void S2::f(int i) {
  (void)[&, &i, &i]{}; // expected-error 2{{'&' cannot precede a capture when the capture default is '&'}}
  (void)[i, i]{ }; // expected-error{{'i' can appear only once in a capture list}}
  (void)[&, i, i]{ }; // expected-error{{'i' can appear only once in a capture list}}
  (void)[] mutable { }; // expected-error{{lambda requires '()' before 'mutable'}}
  (void)[] -> int { }; // expected-error{{lambda requires '()' before return type}}

  delete []() { return new int; }(); // expected-error{{'[]' after delete interpreted as 'delete[]'}}
  delete [] { return new int; }(); // expected-error{{'[]' after delete interpreted as 'delete[]'}}
}

#define bar "bar"
const char *p = "foo"bar; // expected-error {{requires a space between}}
#define ord - '0'
int k = '4'ord; // expected-error {{requires a space between}}

void operator"x" _y(char); // expected-error {{must be '""'}}
void operator L"" _z(char); // expected-error {{encoding prefix}}
void operator "x" "y" U"z" ""_whoops "z" "y"(char); // expected-error {{must be '""'}}

void f() {
  'b'_y;
  'c'_z;
  'd'_whoops;
}

template<typename ...Ts> struct MisplacedEllipsis {
  int a(Ts ...(x)); // expected-error {{'...' must immediately precede declared identifier}}
  int b(Ts ...&x); // expected-error {{'...' must immediately precede declared identifier}}
  int c(Ts ...&); // expected-error {{'...' must be innermost component of anonymous pack declaration}}
  int d(Ts ...(...&...)); // expected-error 2{{'...' must be innermost component of anonymous pack declaration}}
  int e(Ts ...*[]); // expected-error {{'...' must be innermost component of anonymous pack declaration}}
  int f(Ts ...(...*)()); // expected-error 2{{'...' must be innermost component of anonymous pack declaration}}
  int g(Ts ...()); // ok
};
namespace TestMisplacedEllipsisRecovery {
  MisplacedEllipsis<int, char> me;
  int i; char k;
  int *ip; char *kp;
  int ifn(); char kfn();
  int a = me.a(i, k);
  int b = me.b(i, k);
  int c = me.c(i, k);
  int d = me.d(i, k);
  int e = me.e(&ip, &kp);
  int f = me.f(ifn, kfn);
  int g = me.g(ifn, kfn);
}

template<template<typename> ...Foo, // expected-error {{template template parameter requires 'class' after the parameter list}}
         template<template<template<typename>>>> // expected-error 3 {{template template parameter requires 'class' after the parameter list}}
void func();

template<int *ip> struct IP { }; // expected-note{{declared here}}
IP<0> ip0; // expected-error{{null non-type template argument must be cast to template parameter type 'int *'}}

namespace MissingSemi {
  struct a // expected-error {{expected ';' after struct}}
  struct b // expected-error {{expected ';' after struct}}
  enum x : int { x1, x2, x3 } // expected-error {{expected ';' after enum}}
  struct c // expected-error {{expected ';' after struct}}
  enum x : int // expected-error {{expected ';' after enum}}
  // FIXME: The following gives a poor diagnostic (we parse the 'int' and the
  // 'struct' as part of the same enum-base.
  //   enum x : int
  //   struct y
  namespace N {
    struct d // expected-error {{expected ';' after struct}}
  }
}

namespace NonStaticConstexpr {
  struct foo {
    constexpr int i; // expected-error {{non-static data member cannot be constexpr; did you intend to make it const?}}
    constexpr int j = 7; // expected-error {{non-static data member cannot be constexpr; did you intend to make it static?}}
    constexpr const int k; // expected-error {{non-static data member cannot be constexpr; did you intend to make it const?}}
    foo() : i(3), k(4) {
    }
    static int get_j() {
      return j;
    }
  };
}

int RegisterVariable() {
  register int n; // expected-warning {{'register' storage class specifier is deprecated}}
  return n;
}

namespace MisplacedParameterPack {
  template <typename Args...> // expected-error {{'...' must immediately precede declared identifier}}
  void misplacedEllipsisInTypeParameter(Args...);

  template <typename... Args...> // expected-error {{'...' must immediately precede declared identifier}}
  void redundantEllipsisInTypeParameter(Args...);

  template <template <typename> class Args...> // expected-error {{'...' must immediately precede declared identifier}}
  void misplacedEllipsisInTemplateTypeParameter(Args<int>...);

  template <template <typename> class... Args...> // expected-error {{'...' must immediately precede declared identifier}}
  void redundantEllipsisInTemplateTypeParameter(Args<int>...);

  template <int N...> // expected-error {{'...' must immediately precede declared identifier}}
  void misplacedEllipsisInNonTypeTemplateParameter();

  template <int... N...> // expected-error {{'...' must immediately precede declared identifier}}
  void redundantEllipsisInNonTypeTemplateParameter();
}

namespace MisplacedDeclAndRefSpecAfterVirtSpec {
  struct B {
    virtual void f();
    virtual void f() volatile const;
  };
  struct D : B {
    virtual void f() override;
    virtual void f() override final const volatile; // expected-error {{'const' qualifier may not appear after the virtual specifier 'final'}} expected-error {{'volatile' qualifier may not appear after the virtual specifier 'final'}}
  };
  struct B2 {
    virtual void f() &;
    virtual void f() volatile const &&;
  };
  struct D2 : B2 {
    virtual void f() override &; // expected-error {{'&' qualifier may not appear after the virtual specifier 'override'}}
    virtual void f() override final const volatile &&; //  expected-error {{'const' qualifier may not appear after the virtual specifier 'final'}} expected-error {{'volatile' qualifier may not appear after the virtual specifier 'final'}} expected-error {{'&&' qualifier may not appear after the virtual specifier 'final'}}
  };
}
