// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s

using size_t = __SIZE_TYPE__;

// floating-point arguments
template<float> struct Float {};
using F1 = Float<1.0f>; // FIXME expected-error {{sorry}}
using F1 = Float<2.0f / 2>; // FIXME expected-error {{sorry}}

struct S { int n[3]; } s; // expected-note 1+{{here}}
union U { int a, b; } u;
int n; // expected-note 1+{{here}}

// pointers to subobjects
template<int *> struct IntPtr {};
using IPn = IntPtr<&n + 1>; // FIXME expected-error {{refers to subobject}}
using IPn = IntPtr<&n + 1>; // FIXME expected-error {{refers to subobject}}

using IP2 = IntPtr<&s.n[2]>; // FIXME expected-error {{refers to subobject}}
using IP2 = IntPtr<s.n + 2>; // FIXME expected-error {{refers to subobject}}

using IP3 = IntPtr<&s.n[3]>; // FIXME expected-error {{refers to subobject}}
using IP3 = IntPtr<s.n + 3>; // FIXME expected-error {{refers to subobject}}

template<int &> struct IntRef {};
using IPn = IntRef<*(&n + 1)>; // expected-error {{not a constant expression}} expected-note {{dereferenced pointer past the end of 'n'}}
using IPn = IntRef<*(&n + 1)>; // expected-error {{not a constant expression}} expected-note {{dereferenced pointer past the end of 'n'}}

using IP2 = IntRef<s.n[2]>; // FIXME expected-error {{refers to subobject}}
using IP2 = IntRef<*(s.n + 2)>; // FIXME expected-error {{refers to subobject}}

using IP3 = IntRef<s.n[3]>; // expected-error {{not a constant expression}} expected-note {{dereferenced pointer past the end of subobject of 's'}}
using IP3 = IntRef<*(s.n + 3)>; // expected-error {{not a constant expression}} expected-note {{dereferenced pointer past the end of subobject of 's'}}

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
using CI = ComplexInt<1 + 3i>; // FIXME: expected-error {{sorry}}
using CI = ComplexInt<1 + 3i>; // FIXME: expected-error {{sorry}}

template<_Complex float> struct ComplexFloat {};
using CF = ComplexFloat<1.0f + 3.0fi>; // FIXME: expected-error {{sorry}}
using CF = ComplexFloat<1.0f + 3.0fi>; // FIXME: expected-error {{sorry}}

namespace ClassNTTP {
  struct A { // expected-note 2{{candidate}}
    int x, y;
  };
  template<A a> constexpr int f() { return a.y; }
  static_assert(f<A{1,2}>() == 2);

  template<A a> int id;
  constexpr A a = {1, 2};
  static_assert(&id<A{1,2}> == &id<a>);
  static_assert(&id<A{1,3}> != &id<a>);

  int k = id<1>; // expected-error {{no viable conversion from 'int' to 'ClassNTTP::A'}}

  struct B {
    constexpr B() {}
    constexpr B(int) = delete; // expected-note {{here}}
  };
  template<B> struct Q {}; // expected-note {{passing argument to parameter here}}
  Q<1> q; // expected-error {{conversion function from 'int' to 'ClassNTTP::B' invokes a deleted function}}

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
  void f(X<1.0f>) {} // OK, user-defined conversion
  void f(X<2>) {} // expected-error {{conversion from 'int' to 'ConvertedConstant::A' is not allowed in a converted constant expression}}
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
  template<A a> struct X { static_assert(a.n == a.m); }; // expected-error {{static_assert failed due to requirement 'Diags::A{1, 2}.n == Diags::A{1, 2}.m'}}
  template struct X<A{1, 2}>; // expected-note {{in instantiation of template class 'Diags::X<{1, 2}>' requested here}}
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
