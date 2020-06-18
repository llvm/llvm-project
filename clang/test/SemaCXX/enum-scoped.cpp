// RUN: %clang_cc1 -fsyntax-only -pedantic -std=c++11 -verify -triple x86_64-apple-darwin %s

enum class E1 {
  Val1 = 1L
};

enum struct E2 {
  Val1 = '\0'
};

E1 v1 = Val1; // expected-error{{undeclared identifier}}
E1 v2 = E1::Val1;

static_assert(sizeof(E1) == sizeof(int), "bad size");
static_assert(sizeof(E1::Val1) == sizeof(int), "bad size");
static_assert(sizeof(E2) == sizeof(int), "bad size");
static_assert(sizeof(E2::Val1) == sizeof(int), "bad size");

E1 v3 = E2::Val1; // expected-error{{cannot initialize a variable}}
int x1 = E1::Val1; // expected-error{{cannot initialize a variable}}

enum E3 : char {
  Val2 = 1
};

E3 v4 = Val2;
E1 v5 = Val2; // expected-error{{cannot initialize a variable}}

static_assert(sizeof(E3) == 1, "bad size");

int x2 = Val2;

int a1[Val2];
int a2[E1::Val1]; // expected-error{{size of array has non-integer type}}

int* p1 = new int[Val2];
int* p2 = new int[E1::Val1]; // expected-error{{array size expression must have integral or unscoped enumeration type, not 'E1'}}

enum class E4 {
  e1 = -2147483648, // ok
  e2 = 2147483647, // ok
  e3 = 2147483648 // expected-error{{enumerator value evaluates to 2147483648, which cannot be narrowed to type 'int'}}
};

enum class E5 {
  e1 = 2147483647, // ok
  e2 // expected-error{{2147483648 is not representable in the underlying}}
};

enum class E6 : bool {
    e1 = false, e2 = true,
    e3 // expected-error{{2 is not representable in the underlying}}
};

enum E7 : bool {
    e1 = false, e2 = true,
    e3 // expected-error{{2 is not representable in the underlying}}
};

template <class T>
struct X {
  enum E : T {
    e1, e2,
    e3 // expected-error{{2 is not representable in the underlying}}
  };
};

X<bool> X2; // expected-note{{in instantiation of template}}

enum Incomplete1; // expected-error{{C++ forbids forward references}}

enum Complete1 : int;
Complete1 complete1;

enum class Complete2;
Complete2 complete2;

// All the redeclarations below are done twice on purpose. Tests that the type
// of the declaration isn't changed.

enum class Redeclare2; // expected-note{{previous declaration is here}} expected-note{{previous declaration is here}}
enum Redeclare2; // expected-error{{previously declared as scoped}}
enum Redeclare2; // expected-error{{previously declared as scoped}}

enum Redeclare3 : int; // expected-note{{previous declaration is here}} expected-note{{previous declaration is here}}
enum Redeclare3; // expected-error{{previously declared with fixed underlying type}}
enum Redeclare3; // expected-error{{previously declared with fixed underlying type}}

enum class Redeclare5;
enum class Redeclare5 : int; // ok

enum Redeclare6 : int;   // expected-note{{previous declaration is here}} expected-note{{previous declaration is here}}
enum Redeclare6 : short; // expected-error{{redeclared with different underlying type}}
enum Redeclare6 : short; // expected-error{{redeclared with different underlying type}}

enum class Redeclare7;         // expected-note{{previous declaration is here}} expected-note{{previous declaration is here}}
enum class Redeclare7 : short; // expected-error{{redeclared with different underlying type}}
enum class Redeclare7 : short; // expected-error{{redeclared with different underlying type}}

enum : long {
  long_enum_val = 10000
};

enum : long x; // expected-error{{unnamed enumeration must be a definition}} \
// expected-warning{{declaration does not declare anything}}

void PR9333() {
  enum class scoped_enum { yes, no, maybe };
  scoped_enum e = scoped_enum::yes;
  if (e == scoped_enum::no) { }
}

// <rdar://problem/9366066>
namespace rdar9366066 {
  enum class X : unsigned { value };

  void f(X x) {
    x % X::value; // expected-error{{invalid operands to binary expression ('rdar9366066::X' and 'rdar9366066::X')}}
    x % 8; // expected-error{{invalid operands to binary expression ('rdar9366066::X' and 'int')}}
  }
}

// Part 1 of PR10264
namespace test5 {
  namespace ns {
    typedef unsigned Atype;
    enum A : Atype;
  }
  enum ns::A : ns::Atype {
    x, y, z
  };
}

// Part 2 of PR10264
namespace test6 {
  enum A : unsigned;
  struct A::a; // expected-error {{incomplete type 'test6::A' named in nested name specifier}}
  enum A::b; // expected-error {{incomplete type 'test6::A' named in nested name specifier}}
  int A::c; // expected-error {{incomplete type 'test6::A' named in nested name specifier}}
  void A::d(); // expected-error {{incomplete type 'test6::A' named in nested name specifier}}
  void test() {
    (void) A::e; // expected-error {{incomplete type 'test6::A' named in nested name specifier}}
  }
}

namespace PR11484 {
  const int val = 104;
  enum class test1 { owner_dead = val, };
}

namespace N2764 {
  enum class E *x0a; // expected-error {{reference to enumeration must use 'enum' not 'enum class'}}
  enum E2 *x0b; // OK
  enum class E { a, b };
  enum E x1 = E::a; // ok
  enum class E x2 = E::a; // expected-error {{reference to enumeration must use 'enum' not 'enum class'}}

  enum F { a, b };
  enum F y1 = a; // ok
  enum class F y2 = a; // expected-error {{reference to enumeration must use 'enum' not 'enum class'}}

  struct S {
    friend enum class E; // expected-error {{reference to enumeration must use 'enum' not 'enum class'}}
    friend enum class F; // expected-error {{reference to enumeration must use 'enum' not 'enum class'}}

    friend enum G {}; // expected-error {{forward reference}} expected-error {{cannot define a type in a friend declaration}}
    friend enum class H {}; // expected-error {{forward reference}} expected-error {{cannot define a type in a friend declaration}}
    friend enum I : int {}; // expected-error {{forward reference}} expected-error {{cannot define a type in a friend declaration}}

    enum A : int;
    A a;
  } s;

  enum S::A : int {};

  enum class B;
}

enum class N2764::B {};

namespace PR12106 {
  template<typename E> struct Enum {
    Enum() : m_e(E::Last) {}
    E m_e;
  };

  enum eCOLORS { Last };
  Enum<eCOLORS> e;
}

namespace test7 {
  enum class E { e = (struct S*)0 == (struct S*)0 };
  S *p;
}

namespace test8 {
  template<typename T> struct S {
    enum A : int; // expected-note {{here}}
    enum class B; // expected-note {{here}}
    enum class C : int; // expected-note {{here}}
    enum class D : int; // expected-note {{here}}
  };
  template<typename T> enum S<T>::A { a }; // expected-error {{previously declared with fixed underlying type}}
  template<typename T> enum class S<T>::B : char { b }; // expected-error {{redeclared with different underlying}}
  template<typename T> enum S<T>::C : int { c }; // expected-error {{previously declared as scoped}}
  template<typename T> enum class S<T>::D : char { d }; // expected-error {{redeclared with different underlying}}
}

namespace test9 {
  template<typename T> struct S {
    enum class ET : T; // expected-note 2{{here}}
    enum class Eint : int; // expected-note 2{{here}}
  };
  template<> enum class S<int>::ET : int {};
  template<> enum class S<char>::ET : short {}; // expected-error {{different underlying type}}
  template<> enum class S<int>::Eint : short {}; // expected-error {{different underlying type}}
  template<> enum class S<char>::Eint : int {};

  template<typename T> enum class S<T>::ET : int {}; // expected-error {{different underlying type 'int' (was 'short')}}
  template<typename T> enum class S<T>::Eint : T {}; // expected-error {{different underlying type 'short' (was 'int')}}

  // The implicit instantiation of S<short> causes the implicit instantiation of
  // all declarations of member enumerations, so is ill-formed, even though we
  // never instantiate the definitions of S<short>::ET nor S<short>::Eint.
  S<short> s; // expected-note {{in instantiation of}}
}

namespace test10 {
  template<typename T> int f() {
    enum E : int;
    enum E : T; // expected-note {{here}}
    E x;
    enum E : int { e }; // expected-error {{different underlying}}
    x = e;
    return x;
  }
  int k = f<int>();
  int l = f<short>(); // expected-note {{here}}

  template<typename T> int g() {
    enum class E : int;
    enum class E : T; // expected-note {{here}}
    E x;
    enum class E : int { e }; // expected-error {{different underlying}}
    x = E::e;
    return (int)x;
  }
  int m = g<int>();
  int n = g<short>(); // expected-note {{here}}
}

namespace pr13128 {
  // This should compile cleanly
  class C {
    enum class E { C };
  };
}

namespace PR15633 {
  template<typename T> struct A {
    struct B {
      enum class E : T;
      enum class E2 : T;
    };
  };
  template<typename T> enum class A<T>::B::E { e };
  template class A<int>;

  struct B { enum class E; };
  template<typename T> enum class B::E { e }; // expected-error {{enumeration cannot be a template}}
}

namespace PR16900 {
  enum class A;
  A f(A a) { return -a; } // expected-error {{invalid argument type 'PR16900::A' to unary expression}}
}

namespace PR18551 {
  enum class A { A };
  bool f() { return !A::A; } // expected-error {{invalid argument type 'PR18551::A' to unary expression}}
}

namespace rdar15124329 {
  enum class B : bool { F, T };

  const rdar15124329::B T1 = B::T;
  typedef B C;  const C T2 = B::T;

  static_assert(T1 != B::F, "");
  static_assert(T2 == B::T, "");
}

namespace PR18044 {
  enum class E { a };

  int E::e = 0; // expected-error {{does not refer into a class}}
  void E::f() {} // expected-error {{does not refer into a class}}
  struct E::S {}; // expected-error {{no struct named 'S'}}
  struct T : E::S {}; // expected-error {{expected class name}}
  enum E::E {}; // expected-error {{no enum named 'E'}}
  int E::*p; // expected-error {{does not point into a class}}
  using E::f; // expected-error {{no member named 'f'}}

  using E::a; // expected-error {{using declaration cannot refer to a scoped enumerator}}
  E b = a; // expected-error {{undeclared}}
}

namespace test11 {
  enum class E { a };
  typedef E E2;
  E2 f1() { return E::a; }

  bool f() { return !f1(); } // expected-error {{invalid argument type 'test11::E2' (aka 'test11::E') to unary expression}}
}

namespace PR35586 {
  enum C { R, G, B };
  enum B { F = (enum C) -1, T}; // this should compile cleanly, it used to assert.
};
