// RUN: %clang_cc1 -fsyntax-only -std=c++98 -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++14 -verify %s
// RUN: not %clang_cc1 -fsyntax-only -std=c++98 -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck --check-prefix=CXX98 %s
// RUN: not %clang_cc1 -fsyntax-only -std=c++11 -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck --check-prefix=CXX11 %s
// C++0x N2914.

struct X {
  int i;
  static int a;
  enum E { e };
};

using X::i; // expected-error{{using declaration cannot refer to class member}}
using X::s; // expected-error{{using declaration cannot refer to class member}}
using X::e; // expected-error{{using declaration cannot refer to class member}}
using X::E::e; // expected-error{{using declaration cannot refer to class member}} expected-warning 0-1{{C++11}}
#if __cplusplus < 201103L
// expected-note@-3 {{use a const variable}}
// expected-note@-3 {{use a const variable}}
// CXX98-NOT: fix-it:"{{.*}}":{[[@LINE-5]]:
// CXX98-NOT: fix-it:"{{.*}}":{[[@LINE-5]]:
#else
// expected-note@-8 {{use a constexpr variable}}
// expected-note@-8 {{use a constexpr variable}}
// CXX11: fix-it:"{{.*}}":{[[@LINE-10]]:1-[[@LINE-10]]:6}:"constexpr auto e = "
// CXX11: fix-it:"{{.*}}":{[[@LINE-10]]:1-[[@LINE-10]]:6}:"constexpr auto e = "
#endif

void f() {
  using X::i; // expected-error{{using declaration cannot refer to class member}}
  using X::s; // expected-error{{using declaration cannot refer to class member}}
  using X::e; // expected-error{{using declaration cannot refer to class member}}
  using X::E::e; // expected-error{{using declaration cannot refer to class member}} expected-warning 0-1{{C++11}}
#if __cplusplus < 201103L
  // expected-note@-3 {{use a const variable}}
  // expected-note@-3 {{use a const variable}}
  // CXX98-NOT: fix-it:"{{.*}}":{[[@LINE-5]]:
  // CXX98-NOT: fix-it:"{{.*}}":{[[@LINE-5]]:
#else
  // expected-note@-8 {{use a constexpr variable}}
  // expected-note@-8 {{use a constexpr variable}}
  // CXX11: fix-it:"{{.*}}":{[[@LINE-10]]:3-[[@LINE-10]]:8}:"constexpr auto e = "
  // CXX11: fix-it:"{{.*}}":{[[@LINE-10]]:3-[[@LINE-10]]:8}:"constexpr auto e = "
#endif
}

namespace PR21933 {
  struct A { int member; };
  struct B { static int member; };
  enum C { member };

  template <typename T>
  struct X {
    static void StaticFun() {
      using T::member; // expected-error 2{{class member}} expected-note {{use a reference instead}}
#if __cplusplus < 201103L
    // expected-error@-2 {{cannot be used prior to '::'}}
#endif
      (void)member;
    }
  };
  template<typename T>
  struct Y : T { 
    static void StaticFun() {
      using T::member; // expected-error 2{{class member}} expected-note {{use a reference instead}}
      (void)member;
    }
  };

  void f() { 
    X<A>::StaticFun(); // expected-note {{instantiation of}}
    X<B>::StaticFun(); // expected-note {{instantiation of}}
    X<C>::StaticFun();
#if __cplusplus < 201103L
    // expected-note@-2 {{instantiation of}}
#endif
    Y<A>::StaticFun(); // expected-note {{instantiation of}}
    Y<B>::StaticFun(); // expected-note {{instantiation of}}
  }

  template<typename T, typename U> void value_vs_value() {
    using T::a; // expected-note {{previous}}
#if __cplusplus < 201103L
    // expected-error@-2 {{cannot be used prior to '::'}}
#endif
    extern int a(); // expected-error {{different kind of symbol}}
    a();

    extern int b(); // expected-note {{previous}}
    using T::b; // expected-error {{different kind of symbol}}
    b();

    using T::c; // expected-note {{previous}}
    using U::c; // expected-error-re {{redefinition of 'c'{{$}}}}
    c();
  }

  template<typename T, typename U> void value_vs_type() {
    using T::Xt; // expected-note {{previous}}
    typedef struct {} Xt; // expected-error {{different kind of symbol}}
    (void)Xt;

    using T::Xs; // expected-note {{hidden by}}
    struct Xs {};
    (void)Xs;
    Xs xs; // expected-error {{must use 'struct'}}

    using T::Xe; // expected-note {{hidden by}}
    enum Xe {};
    (void)Xe;
    Xe xe; // expected-error {{must use 'enum'}}

    typedef struct {} Yt; // expected-note {{candidate}}
    using T::Yt; // eypected-error {{different kind of symbol}} expected-note {{candidate}}
    Yt yt; // expected-error {{ambiguous}}

    struct Ys {};
    using T::Ys; // expected-note {{hidden by}}
    (void)Ys;
    Ys ys; // expected-error {{must use 'struct'}}

    enum Ye {};
    using T::Ye; // expected-note {{hidden by}}
    Ye ye; // expected-error {{must use 'enum'}}
  }

  template<typename T> void type() {
    // Must be a class member because T:: can only name a class or enum,
    // and an enum cannot have a type member.
    using typename T::X; // expected-error {{cannot refer to class member}}
  }

  namespace N1 { enum E { a, b, c }; }
  namespace N2 { enum E { a, b, c }; }
  void g() { value_vs_value<N1::E, N2::E>(); }
#if __cplusplus < 201103L
    // expected-note@-2 {{in instantiation of}}
#endif

#if __cplusplus >= 201402L
  namespace partial_substitute {
    template<typename T> auto f() {
      return [](auto x) {
        using A = typename T::template U<decltype(x)>;
        using A::E::e;
        struct S : A {
          using A::f;
          using typename A::type;
          type f(int) { return e; }
        };
        return S();
      };
    }
    enum Enum { e };
    struct X {
      template<typename T> struct U {
        int f(int, int);
        using type = int;
        using E = Enum;
      };
    };
    int test() {
      auto s = f<X>()(0);
      return s.f(0) + s.f(0, 0);
    }

    template<typename T, typename U> auto g() {
      return [](auto x) {
        using X = decltype(x);
        struct S : T::template Q<X>, U::template Q<X> {
          using T::template Q<X>::f;
          using U::template Q<X>::f;
          void h() { f(); }
          void h(int n) { f(n); }
        };
        return S();
      };
    }
    struct A { template<typename> struct Q { int f(); }; };
    struct B { template<typename> struct Q { int f(int); }; };
    int test2() {
      auto s = g<A, B>()(0);
      s.f();
      s.f(0);
      s.h();
      s.h(0);
    }
  }
#endif

  template<typename T, typename U> struct RepeatedMember : T, U {
    // FIXME: This is the wrong error: we should complain that a member type
    // cannot be redeclared at class scope.
    using typename T::type; // expected-note {{candidate}}
    using typename U::type; // expected-note {{candidate}}
    type x; // expected-error {{ambiguous}}
  };
}

struct S {
  static int n;
  struct Q {};
  enum E {};
  typedef Q T;
  void f();
  static void g();
};

using S::n; // expected-error{{class member}} expected-note {{use a reference instead}}
#if __cplusplus < 201103L
// CXX98-NOT: fix-it:"{{.*}}":{[[@LINE-2]]
#else
// CXX11: fix-it:"{{.*}}":{[[@LINE-4]]:1-[[@LINE-4]]:6}:"auto &n = "
#endif

using S::Q; // expected-error{{class member}}
#if __cplusplus < 201103L
// expected-note@-2 {{use a typedef declaration instead}}
// CXX98: fix-it:"{{.*}}":{[[@LINE-3]]:1-[[@LINE-3]]:6}:"typedef"
// CXX98: fix-it:"{{.*}}":{[[@LINE-4]]:11-[[@LINE-4]]:11}:" Q"
#else
// expected-note@-6 {{use an alias declaration instead}}
// CXX11: fix-it:"{{.*}}":{[[@LINE-7]]:7-[[@LINE-7]]:7}:"Q = "
#endif

using S::E; // expected-error{{class member}}
#if __cplusplus < 201103L
// expected-note@-2 {{use a typedef declaration instead}}
// CXX98: fix-it:"{{.*}}":{[[@LINE-3]]:1-[[@LINE-3]]:6}:"typedef"
// CXX98: fix-it:"{{.*}}":{[[@LINE-4]]:11-[[@LINE-4]]:11}:" E"
#else
// expected-note@-6 {{use an alias declaration instead}}
// CXX11: fix-it:"{{.*}}":{[[@LINE-7]]:7-[[@LINE-7]]:7}:"E = "
#endif

using S::T; // expected-error{{class member}}
#if __cplusplus < 201103L
// expected-note@-2 {{use a typedef declaration instead}}
// CXX98: fix-it:"{{.*}}":{[[@LINE-3]]:1-[[@LINE-3]]:6}:"typedef"
// CXX98: fix-it:"{{.*}}":{[[@LINE-4]]:11-[[@LINE-4]]:11}:" T"
#else
// expected-note@-6 {{use an alias declaration instead}}
// CXX11: fix-it:"{{.*}}":{[[@LINE-7]]:7-[[@LINE-7]]:7}:"T = "
#endif

using S::f; // expected-error{{class member}}
using S::g; // expected-error{{class member}}

void h() {
  using S::n; // expected-error{{class member}} expected-note {{use a reference instead}}
#if __cplusplus < 201103L
  // CXX98-NOT: fix-it:"{{.*}}":{[[@LINE-2]]
#else
  // CXX11: fix-it:"{{.*}}":{[[@LINE-4]]:3-[[@LINE-4]]:8}:"auto &n = "
#endif

  using S::Q; // expected-error{{class member}}
#if __cplusplus < 201103L
  // expected-note@-2 {{use a typedef declaration instead}}
  // CXX98: fix-it:"{{.*}}":{[[@LINE-3]]:3-[[@LINE-3]]:8}:"typedef"
  // CXX98: fix-it:"{{.*}}":{[[@LINE-4]]:13-[[@LINE-4]]:13}:" Q"
#else
  // expected-note@-6 {{use an alias declaration instead}}
  // CXX11: fix-it:"{{.*}}":{[[@LINE-7]]:9-[[@LINE-7]]:9}:"Q = "
#endif

  using S::E; // expected-error{{class member}}
#if __cplusplus < 201103L
  // expected-note@-2 {{use a typedef declaration instead}}
  // CXX98: fix-it:"{{.*}}":{[[@LINE-3]]:3-[[@LINE-3]]:8}:"typedef"
  // CXX98: fix-it:"{{.*}}":{[[@LINE-4]]:13-[[@LINE-4]]:13}:" E"
#else
  // expected-note@-6 {{use an alias declaration instead}}
  // CXX11: fix-it:"{{.*}}":{[[@LINE-7]]:9-[[@LINE-7]]:9}:"E = "
#endif

  using S::T; // expected-error{{class member}}
#if __cplusplus < 201103L
  // expected-note@-2 {{use a typedef declaration instead}}
  // CXX98: fix-it:"{{.*}}":{[[@LINE-3]]:3-[[@LINE-3]]:8}:"typedef"
  // CXX98: fix-it:"{{.*}}":{[[@LINE-4]]:13-[[@LINE-4]]:13}:" T"
#else
  // expected-note@-6 {{use an alias declaration instead}}
  // CXX11: fix-it:"{{.*}}":{[[@LINE-7]]:9-[[@LINE-7]]:9}:"T = "
#endif

  using S::f; // expected-error{{class member}}
  using S::g; // expected-error{{class member}}
}
