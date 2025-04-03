// RUN: %clang_cc1 -std=c++2b -fsyntax-only -verify %s

enum class Z;

struct bar {};

using Int = int;
using Float = float;
using Bar = bar;

namespace t1 {
template <class A1> struct A {
  static constexpr A1 a = {};
};

Z x1 = A<Int>::a;
// expected-error@-1 {{with an lvalue of type 'const Int' (aka 'const int')}}
} // namespace t1

namespace t2 {
template <class A1, class A2> struct A {
  static constexpr A1 A2::*a = {};
};

Z x1 = A<Int, Bar>::a;
// expected-error@-1 {{with an lvalue of type 'Int Bar::*const'}}
} // namespace t2

namespace t3 {
template <class A1> struct A {
  template <class B1> struct B {
    static constexpr A1 B1::*a = {};
  };
};

Z x1 = A<Float>::B<Bar>::a;
// expected-error@-1 {{with an lvalue of type 'Float Bar::*const'}}
} // namespace t3

namespace t4 {
template <class A1> A1 (*a) ();

// FIXME: resugar this
Z x1 = decltype(a<Int>){}();
// expected-error@-1 {{with an rvalue of type 'int'}}
} // namespace t4

namespace t5 {
template <class A1> struct A {
  A1(*a)
  ();
};

// FIXME: resugar this
Z x1 = decltype(A<Int>().a){}();
// expected-error@-1 {{with an rvalue of type 'int'}}
} // namespace t5

namespace t6 {
template <class A1, class A2> struct A { A2 A1::*f(); };

using M = int;
using N = int;

struct B {};
using X = B;
using Y = B;

auto a = &A<X, M>::f;
Z x1 = a;
// expected-error@-1 {{with an lvalue of type 'M X::*(A<X, M>::*)()'}}

A<Y, N> b;
Z x2 = (b.*a)();
// expected-error@-1 {{with an rvalue of type 'M X::*'}}

Z x3 = decltype((b.*a)()){};
// expected-error@-1 {{with an rvalue of type 'decltype((b .* a)())' (aka 'M X::*')}}
} // namespace t6

namespace t7 {
template <class A1> struct A { A1 a; };
auto [a] = A<Int>{};

Z x1 = a;
// expected-error@-1 {{with an lvalue of type 'Int' (aka 'int')}}
} // namespace t7

namespace t8 {
template <class A1> struct A {
  template <class B1> static constexpr B1 (*b)(A1) = nullptr;
};

Z x1 = A<Float>::b<Int>;
// expected-error@-1 {{with an lvalue of type 'Int (*const)(Float)' (aka 'int (*const)(float)')}}
} // namespace t8

namespace t9 {
template <class A1> struct A {
  template <class B1> static constexpr auto b = (B1(*)(A1)){};
};

Z x1 = A<Float>::b<Int>;
// expected-error@-1 {{with an lvalue of type 'Int (*const)(Float)' (aka 'int (*const)(float)'}}
} // namespace t9

namespace t10 {
template <class A1> struct A {
  template <class B1> static constexpr A1 (*m)(B1) = nullptr;
};

Z x1 = A<Int>().template m<Float>;
// expected-error@-1 {{with an lvalue of type 'Int (*const)(Float)' (aka 'int (*const)(float)'}}
} // namespace t10

namespace t11 {
template <class A1> A1 a;
template <class A2> A2 a<A2 *>;

Z x1 = a<Int>;
// expected-error@-1 {{with an lvalue of type 'Int' (aka 'int')}}

Z x2 = a<Float *>;
// expected-error@-1 {{with an lvalue of type 'Float' (aka 'float')}}
} // namespace t11

namespace t12 {
template<class A1> A1 *a;
template<int A3, class A4> decltype(a<A4[A3 - 1]>) a<A4[A3]>;

// FIXME: resugar this
Z x1 = *a<Int[1]>;
// expected-error@-1 {{with an lvalue of type 'int[0]'}}
} // namespace t12

namespace t13 {
template <class A1> struct A { A1 foo(); };

Z x1 = A<Int>().foo();
// expected-error@-1 {{with an rvalue of type 'Int' (aka 'int')}}
} // namespace t13

namespace t14 {
template <class A1> struct A {
  auto foo() { return A1(); };
};

Z x1 = A<Int>().foo();
// expected-error@-1 {{with an rvalue of type 'Int' (aka 'int')}}
} // namespace t14

namespace t15 {
template <class A1> struct A {
  template <class B1> auto foo1() -> A1 (*)(B1);
  template <class B1> auto foo2(B1) -> A1 (*)(B1);
};

Z x1 = A<Int>().foo1<Float>();
// expected-error@-1 {{with an rvalue of type 'Int (*)(Float)' (aka 'int (*)(float)'}}

Z x2 = A<Int>().foo2(Float());
// expected-error@-1 {{with an rvalue of type 'Int (*)(Float)' (aka 'int (*)(float)'}}
} // namespace t15

namespace t16 {
template <class A1> struct A {
  static auto foo() -> A1;
};

Z x1 = A<Int>().foo();
// expected-error@-1 {{with an rvalue of type 'Int' (aka 'int')}}
} // namespace t16

namespace t17 {
template <class A1> static auto foo() -> A1;

Z x1 = foo<Int>();
// expected-error@-1 {{with an rvalue of type 'Int' (aka 'int')}}
} // namespace t17

namespace t18 {
template <class A1> static auto foo(A1) -> A1*;

Z x1 = foo(Int());
// expected-error@-1 {{with an rvalue of type 'Int *' (aka 'int *')}}
} // namespace t18

namespace t19 {
template <class A1> struct A {
  template <class B1> static auto foo() -> A1 (*)(B1);
};

Z x1 = A<Int>().template foo<Float>();
// expected-error@-1 {{with an rvalue of type 'Int (*)(Float)' (aka 'int (*)(float)'}}

Z x2 = A<Int>::template foo<Float>();
// expected-error@-1 {{with an rvalue of type 'Int (*)(Float)' (aka 'int (*)(float)'}}
} // namespace t19

namespace t20 {
template <class A1> struct A {
  A1 m;
};

Z x1 = A<Int>().m;
// expected-error@-1 {{with an rvalue of type 'Int' (aka 'int')}}
} // namespace t20

namespace t21 {
template <class A1> struct A {
  static A1 m;
};

Z x1 = A<Int>().m;
// expected-error@-1 {{with an lvalue of type 'Int' (aka 'int')}}
} // namespace t21

namespace t22 {
template <class A1> struct A {
  struct {
    A1 m;
  };
};

Z x1 = A<Int>().m;
// expected-error@-1 {{with an rvalue of type 'Int' (aka 'int')}}
} // namespace t22
