// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s

using size_t = __SIZE_TYPE__;

namespace std {
  struct type_info;
}

// floating-point arguments
template<float> struct Float {};
using F1 = Float<1.0f>;
using F1 = Float<2.0f / 2>;

struct S { int n[3]; } s; // expected-note 1+{{here}}
union U { int a, b; } u;
int n; // expected-note 1+{{here}}

// pointers to subobjects
template<int *> struct IntPtr {};
using IPn = IntPtr<&n + 1>;
using IPn = IntPtr<&n + 1>;

using IPn2 = IntPtr<&n + 2>; // expected-error {{not a constant expression}} expected-note {{cannot refer to element 2 of non-array object}}

using IP2 = IntPtr<&s.n[2]>;
using IP2 = IntPtr<s.n + 2>;

using IP3 = IntPtr<&s.n[3]>;
using IP3 = IntPtr<s.n + 3>;

using IP5 = IntPtr<&s.n[5]>; // expected-error {{not a constant expression}} expected-note {{cannot refer to element 5 of array of 3 elements}}

template<int &> struct IntRef {};
using IRn = IntRef<*(&n + 1)>; // expected-error {{not a constant expression}} expected-note {{dereferenced pointer past the end of 'n'}}
using IRn = IntRef<*(&n + 1)>; // expected-error {{not a constant expression}} expected-note {{dereferenced pointer past the end of 'n'}}

using IR2 = IntRef<s.n[2]>;
using IR2 = IntRef<*(s.n + 2)>;

using IR3 = IntRef<s.n[3]>; // expected-error {{not a constant expression}} expected-note {{dereferenced pointer past the end of subobject of 's'}}
using IR3 = IntRef<*(s.n + 3)>; // expected-error {{not a constant expression}} expected-note {{dereferenced pointer past the end of subobject of 's'}}

// classes
template<S> struct Struct {};
using S123 = Struct<S{1, 2, 3}>;
using S123 = Struct<S{1, 2, 3}>; // expected-note {{previous}}
using S123 = Struct<S{1, 2, 4}>; // expected-error {{different types}}
template<U> struct Union {};
using U1 = Union<U{1}>;
using U1 = Union<U{.a = 1}>; // expected-note {{previous}}
using U1 = Union<U{.b = 1}>; // expected-error {{different types}}

// miscellaneous scalar types
template<_Complex int> struct ComplexInt {};
using CI = ComplexInt<1 + 3i>;
using CI = ComplexInt<3i + 1>;

template<_Complex float> struct ComplexFloat {};
using CF = ComplexFloat<1.0f + 3.0fi>;
using CF = ComplexFloat<3.0fi + 1.0f>;

namespace ClassNTTP {
  struct A { // expected-note 2{{candidate}}
    int x, y;
  };
  template<A a> constexpr int f() { return a.y; }
  static_assert(f<A{1,2}>() == 2);

  template<A a> int id; // #ClassNTTP1
  constexpr A a = {1, 2};
  static_assert(&id<A{1,2}> == &id<a>);
  static_assert(&id<A{1,3}> != &id<a>);

  int k = id<1>; // expected-error {{no viable conversion from 'int' to 'A'}}
                 // expected-note@#ClassNTTP1 {{passing argument to parameter 'a' here}}

  struct B {
    constexpr B() {}
    constexpr B(int) = delete; // expected-note {{here}}
  };
  template<B> struct Q {}; // expected-note {{passing argument to parameter here}}
  Q<1> q; // expected-error {{conversion function from 'int' to 'B' invokes a deleted function}}

  struct C {
    constexpr C() {}
    C(const C&) = delete; // expected-note {{here}}
  };
  template<C> struct R {}; // expected-note {{passing argument to parameter here}}
  constexpr C c;
  R<c> r; // expected-error {{call to deleted constructor}}
}

namespace ConvertedConstant {
  struct A {
    constexpr A(float) {}
  };
  template <A> struct X {};
  void f(X<1.0f>) {}
  void g(X<2>) {}

  struct {
    int i : 2;
  } b;
  template <const int&> struct Y {};
  void f(Y<b.i>) {} // expected-error {{reference cannot bind to bit-field in converted constant expression}}
}

namespace CopyCounting {
  // Make sure we don't use the copy constructor when transferring the "same"
  // template parameter object around.
  struct A { int n; constexpr A(int n = 0) : n(n) {} constexpr A(const A &a) : n(a.n+1) {} };
  template<A a> struct X {};
  template<A a> constexpr int f(X<a> x) { return a.n; }

  static_assert(f(X<A{}>()) == 0);

  template<A a> struct Y { void f(); };
  template<A a> void g(Y<a> y) { y.Y<a>::f(); }
  void h() { constexpr A a; g<a>(Y<a>{}); }

  template<A a> struct Z {
    constexpr int f() {
      constexpr A v = a; // this is {a.n+1}
      return Z<v>().f() + 1; // this is Z<{a.n+2}>
    }
  };
  template<> struct Z<A{20}> {
    constexpr int f() {
      return 32;
    }
  };
  static_assert(Z<A{}>().f() == 42);
}

namespace StableAddress {
  template<size_t N> struct str {
    char arr[N];
  };
  // FIXME: Deduction guide not needed with P1816R0.
  template<size_t N> str(const char (&)[N]) -> str<N>;

  template<str s> constexpr int sum() {
    int n = 0;
    for (char c : s.arr)
      n += c;
    return n;
  }
  static_assert(sum<str{"$hello $world."}>() == 1234);
}

namespace TemplateSpecializations {
  struct A { int arr[10]; };
  template<A> struct X; // expected-note {{here}}

  using T = X<A{1, 2, 3}>;
  using T = X<A{1, 2, 3, 0}>;
  using T = X<A{1, 2, 3, 0, 0}>;
  using T = X<A{1, 2, 3, 0, 0, 0}>;

  template<> struct X<A{1, 2, 3, 4}> {};
  X<A{1, 2, 3, 4, 0}> x;

  template<auto V, auto W> constexpr bool Same = false;
  template<auto V> constexpr bool Same<V, V> = true;
  static_assert(Same<A{}, A{0, 0}>);
  static_assert(Same<A{1}, A{1, 0}>);
  static_assert(!Same<A{1}, A{1, 1}>);

  // We can't directly specialize on member values...
  template<int N> // expected-note {{parameter 'N'}}
    struct X<A{N, N}> {}; // expected-error {{cannot be deduced}}

  // ... but we can fake it up.
  // template<int N> struct X<A{N, N}>
  template <A V> requires Same<V, A{V.arr[0], V.arr[0]}>
  struct X<V> {
    static constexpr bool match = true;
  };
  static_assert(X<A{1, 1}>::match);
  static_assert(X<A{2, 2}>::match);
  static_assert(X<A{1, 2}>::match); // expected-error {{undefined}}

  template<int, A> struct Y; // expected-note {{here}}
  template<int N> struct Y<N, A{N, N, N}> {};
  Y<1, A{1, 1, 1, 0}> y1;
  Y<1, A{1, 1, 1, 1}> y2; // expected-error {{undefined}}

  template<A, A> struct Z; // expected-note {{here}}
  template<A V> struct Z<V, V> {};
  Z<A{1, 2}, A{1, 2, 0}> z1;
  Z<A{1, 2}, A{1, 3}> z2; // expected-error {{undefined}}

  template struct Z<A{1}, A{1, 0}>;
}

namespace Diags {
  struct A { int n, m; };
  template<A a> struct X { static_assert(a.n == a.m); }; // expected-error {{static assertion failed due to requirement 'Diags::A{1, 2}.n == Diags::A{1, 2}.m'}} \
                                                         // expected-note {{evaluates to '1 == 2'}}
  template struct X<A{1, 2}>; // expected-note {{in instantiation of template class 'Diags::X<A{1, 2}>' requested here}}
}

namespace CTADPartialOrder {
  template<int> struct A {};
  template<typename T, typename U, A a> struct X; // expected-note {{declared here}}
  template<typename T, A a> struct X<T, int, a> { static constexpr int n = 1; }; // expected-note {{matches}}
  template<typename T, A a> struct X<T *, int, a> { static constexpr int n = 2; };
  template<typename T, A a> struct X<T, T, a> { static constexpr int n = 3; }; // expected-note {{matches}}

  A<0> a;
  static_assert(X<void, int, a>::n == 1);
  static_assert(X<int*, int, a>::n == 2);
  static_assert(X<void, void, a>::n == 3);
  static_assert(X<int, int, a>::n == -1); // expected-error {{ambiguous}}
  static_assert(X<int*, void, a>::n == 2); // expected-error {{undefined}}

  template<typename T, A<0> a> struct X<T, T, a> { static constexpr int n = 4; };
  static_assert(X<float, float, a>::n == 4);
}

namespace UnnamedBitfield {
  struct A {
    __INT32_TYPE__ : 32;
  };
  // Make sure we don't distinguish between the unnamed bit-field being
  // uninitialized and it being zeroed. Those are not distinct states
  // according to [temp.type]p2.
  //
  // FIXME: We shouldn't track a value for unnamed bit-fields, nor number
  // them when computing field indexes.
  template <A> struct X {};
  constexpr A a;
  using T = X<a>;
  using T = X<A{}>;
  using T = X<(A())>;
  // Once we support bit-casts involving bit-fields, this should be valid too.
  using T = X<__builtin_bit_cast(A, 0)>; // expected-error {{constant}} expected-note {{not yet supported}}
}

namespace Temporary {
  template<const int &> struct A {};
  A<0> a0; // expected-error {{conversion from 'int' to 'const int &' in converted constant expression would bind reference to a temporary}}

  A<(const int&)1> a1; // expected-error {{reference to temporary object is not allowed in a template argument}}
  A<(int&&)2> a2; // expected-error {{reference to temporary object is not allowed in a template argument}}

  // FIXME: There's really no good reason to reject these cases.
  int &&r3 = 3;
  const int &r4 = 4;
  A<r3> a3; // expected-error {{reference to temporary object is not allowed in a template argument}}
  A<r4> a4; // expected-error {{reference to temporary object is not allowed in a template argument}}

  struct X { int a[5]; };
  X &&x = X{};
  A<x.a[3]> a5; // expected-error {{reference to subobject of temporary object}}

  template<const int*> struct B {};
  B<&(int&)(int&&)0> b0; // expected-error {{pointer to temporary object}}
  B<&r3> b3; // expected-error {{pointer to temporary object}}
  B<&x.a[3]> b5; // expected-error {{pointer to subobject of temporary object}}

  struct C { const int *p[2]; };
  template<C> struct D {};
  D<C{nullptr, &r3}> d; // expected-error {{pointer to temporary object}}
}

namespace StringLiteral {
  template<decltype(auto)> struct Y {};
  Y<&"hello"> y1; // expected-error {{pointer to string literal}}
  Y<"hello"> y2; // expected-error {{reference to string literal}}
  Y<+"hello"> y3; // expected-error {{pointer to subobject of string literal}}
  Y<"hello"[2]> y4; // expected-error {{reference to subobject of string literal}}

  struct A { const char *p; };
  struct B { const char &r; };
  Y<A{"hello"}> y5; // expected-error {{pointer to subobject of string literal}}
  Y<B{"hello"[2]}> y6; // expected-error {{reference to subobject of string literal}}
}

namespace TypeInfo {
  template<decltype(auto)> struct Y {};
  Y<&typeid(int)> y1; // expected-error {{pointer to type_info object}}
  Y<typeid(int)> y2; // expected-error {{reference to type_info object}}

  struct A { const std::type_info *p; };
  struct B { const std::type_info &r; };
  Y<A{&typeid(int)}> y3; // expected-error {{pointer to type_info object}}
  Y<B{typeid(int)}> y4; // expected-error {{reference to type_info object}}
}

namespace Predefined {
  template<decltype(auto)> struct Y {};

  struct A { const char *p; };
  struct B { const char &r; };
  void f() {
    // decltype(__func__) is an array, which decays to a pointer parameter.
    Y<__func__>(); // expected-error {{pointer to subobject of predefined '__func__' variable}}
    Y<__PRETTY_FUNCTION__>(); // expected-error {{pointer to subobject}}
    Y<(__func__)>(); // expected-error {{reference to predefined '__func__' variable}}
    Y<&__func__>(); // expected-error {{pointer to predefined '__func__' variable}}
    Y<*&__func__>(); // expected-error {{reference to predefined '__func__' variable}}
    Y<A{__func__}>(); // expected-error {{pointer to subobject of predefined '__func__' variable}}
    Y<B{__func__[0]}>(); // expected-error {{reference to subobject of predefined '__func__' variable}}
  }
}

namespace DependentCTAD {
  template<auto> struct A {};
  template<template<typename> typename T, T V> void f(A<V>); // expected-note {{couldn't infer template argument 'T'}}
  template<typename T> struct B { constexpr B(T) {} };

  void g() {
    // PR50039: Note that we could in principle deduce T here, but the language
    // deduction rules don't support that.
    f(A<B(0)>()); // expected-error {{no matching function}}
    f<B>(A<B(0)>()); // OK
  }
}

namespace GH48731 {
template <int> using N = int;
struct X { template<typename T> void f(); };
template<int ...Is> decltype((X().f<N<Is>>(), ...)) x;
template<int ...Is> decltype(((new X())->f<N<Is>>(), ...)) y;

struct A {};
template<int> using Tfoo = A;
template<int ...Ns> void foo(A a) {
  (a.~Tfoo<Ns>(), ...);
}

struct B { operator int(); };
template<int> using Tbar = int;
template<int ...Ns> void bar(B b) {
  (b.operator Tbar<Ns>(), ...);
}
}
