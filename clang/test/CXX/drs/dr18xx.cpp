// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -verify=expected,cxx98 -fexceptions -Wno-deprecated-builtins -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -verify=expected,cxx11-17,since-cxx11 -fexceptions -Wno-deprecated-builtins -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -verify=expected,cxx11-17,since-cxx11,since-cxx14 -fexceptions -Wno-deprecated-builtins -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown %s -verify=expected,cxx11-17,since-cxx11,since-cxx14 -fexceptions -Wno-deprecated-builtins -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx20,since-cxx11,since-cxx14 -fexceptions -Wno-deprecated-builtins -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx20,since-cxx11,since-cxx14 -fexceptions -Wno-deprecated-builtins -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++2c -triple x86_64-unknown-unknown %s -verify=expected,since-cxx20,since-cxx11,since-cxx14 -fexceptions -Wno-deprecated-builtins -fcxx-exceptions -pedantic-errors

#if __cplusplus == 199711L
#define static_assert(...) __extension__ _Static_assert(__VA_ARGS__)
// cxx98-error@-1 {{variadic macros are a C99 feature}}
#endif

namespace dr1812 { // dr1812: no
                   // NB: dup 1710
#if __cplusplus >= 201103L
template <typename T> struct A {
  using B = typename T::C<int>;
  // since-cxx11-error@-1 {{use 'template' keyword to treat 'C' as a dependent template name}}
};
#endif
} // namespace dr1812

namespace dr1813 { // dr1813: 7
  struct B { int i; };
  struct C : B {};
  struct D : C {};
  struct E : D { char : 4; };

  static_assert(__is_standard_layout(B), "");
  static_assert(__is_standard_layout(C), "");
  static_assert(__is_standard_layout(D), "");
  static_assert(!__is_standard_layout(E), "");

  struct Q {};
  struct S : Q {};
  struct T : Q {};
  struct U : S, T {};

  static_assert(__is_standard_layout(Q), "");
  static_assert(__is_standard_layout(S), "");
  static_assert(__is_standard_layout(T), "");
  static_assert(!__is_standard_layout(U), "");
}

namespace dr1814 { // dr1814: yes
#if __cplusplus >= 201103L
  void test() {
    auto lam = [](int x = 42) { return x; };
  }
#endif
}

namespace dr1815 { // dr1815: no
#if __cplusplus >= 201402L
  // FIXME: needs codegen test
  struct A { int &&r = 0; }; // #dr1815-A 
  A a = {};
  // since-cxx14-warning@-1 {{sorry, lifetime extension of temporary created by aggregate initialization using default member initializer is not supported; lifetime of temporary will end at the end of the full-expression}} FIXME
  //   since-cxx14-note@#dr1815-A {{initializing field 'r' with default member initializer}}

  struct B { int &&r = 0; }; // #dr1815-B
  // since-cxx14-error@-1 {{reference member 'r' binds to a temporary object whose lifetime would be shorter than the lifetime of the constructed object}}
  //   since-cxx14-note@#dr1815-B {{initializing field 'r' with default member initializer}}
  //   since-cxx14-note@#dr1815-b {{in implicit default constructor for 'dr1815::B' first required here}}
  B b; // #dr1815-b
#endif
}

namespace dr1821 { // dr1821: 2.9
struct A {
  template <typename> struct B {
    void f();
  };
  template <typename T> void B<T>::f(){};
  // expected-error@-1 {{non-friend class member 'f' cannot have a qualified name}}

  struct C {
    void f();
  };
  void C::f() {}
  // expected-error@-1 {{non-friend class member 'f' cannot have a qualified name}}
};
} // namespace dr1821

namespace dr1822 { // dr1822: yes
#if __cplusplus >= 201103L
  double a;
  auto x = [] (int a) {
    static_assert(__is_same(decltype(a), int), "should be resolved to lambda parameter");
  };
#endif
}

namespace dr1837 { // dr1837: 3.3
#if __cplusplus >= 201103L
  template <typename T>
  struct Fish { static const bool value = true; };

  struct Other {
    int p();
    auto q() -> decltype(p()) *;
  };

  class Outer {
    friend auto Other::q() -> decltype(this->p()) *;
    // since-cxx11-error@-1 {{invalid use of 'this' outside of a non-static member function}}
    int g();
    int f() {
      extern void f(decltype(this->g()) *);
      struct Inner {
        static_assert(Fish<decltype(this->g())>::value, "");
        // since-cxx11-error@-1 {{invalid use of 'this' outside of a non-static member function}}
        enum { X = Fish<decltype(this->f())>::value };
        // since-cxx11-error@-1 {{invalid use of 'this' outside of a non-static member function}}
        struct Inner2 : Fish<decltype(this->g())> { };
        // since-cxx11-error@-1 {{invalid use of 'this' outside of a non-static member function}}
        friend void f(decltype(this->g()) *);
        // since-cxx11-error@-1 {{invalid use of 'this' outside of a non-static member function}}
        friend auto Other::q() -> decltype(this->p()) *;
        // since-cxx11-error@-1 {{invalid use of 'this' outside of a non-static member function}}
      };
      return 0;
    }
  };

  struct A {
    int f();
    bool b = [] {
      struct Local {
        static_assert(sizeof(this->f()) == sizeof(int), "");
      };
    };
  };
#endif
}

namespace dr1872 { // dr1872: 9
#if __cplusplus >= 201103L
  template<typename T> struct A : T {
    constexpr int f() const { return 0; }
  };
  struct X {};
  struct Y { virtual int f() const; };
  struct Z : virtual X {};

  constexpr int x = A<X>().f();
  constexpr int y = A<Y>().f();
  // cxx11-17-error@-1 {{constexpr variable 'y' must be initialized by a constant expression}}
  //   cxx11-17-note@-2 {{cannot evaluate call to virtual function in a constant expression in C++ standards before C++20}}
#if __cplusplus >= 202002L
  static_assert(y == 0);
#endif
  // Note, this is invalid even though it would not use virtual dispatch.
  constexpr int y2 = A<Y>().A<Y>::f();
  // cxx11-17-error@-1 {{constexpr variable 'y2' must be initialized by a constant expression}}
  //   cxx11-17-note@-2 {{cannot evaluate call to virtual function in a constant expression in C++ standards before C++20}}
#if __cplusplus >= 202002L
  static_assert(y == 0);
#endif
  constexpr int z = A<Z>().f();
  // since-cxx11-error@-1 {{constexpr variable 'z' must be initialized by a constant expression}}
  //   since-cxx11-note@-2 {{non-literal type 'A<Z>' cannot be used in a constant expression}}
#endif
}

namespace dr1881 { // dr1881: 7
  struct A { int a : 4; };
  struct B : A { int b : 3; };
  static_assert(__is_standard_layout(A), "");
  static_assert(!__is_standard_layout(B), "");

  struct C { int : 0; };
  struct D : C { int : 0; };
  static_assert(__is_standard_layout(C), "");
  static_assert(!__is_standard_layout(D), "");
}

void dr1891() { // dr1891: 4
#if __cplusplus >= 201103L
  int n;
  auto a = []{}; // #dr1891-a
  auto b = [=]{ return n; }; // #dr1891-b
  typedef decltype(a) A;
  typedef decltype(b) B;

  static_assert(!__has_trivial_constructor(A), "");
  // since-cxx20-error@-1 {{failed}}
  static_assert(!__has_trivial_constructor(B), "");

  // C++20 allows default construction for non-capturing lambdas (P0624R2).
  A x;
  // cxx11-17-error@-1 {{no matching constructor for initialization of 'A' (aka '(lambda at}}
  //   cxx11-17-note@#dr1891-a {{candidate constructor (the implicit copy constructor) not viable: requires 1 argument, but 0 were provided}}
  //   cxx11-17-note@#dr1891-a {{candidate constructor (the implicit move constructor) not viable: requires 1 argument, but 0 were provided}}
  B y;
  // since-cxx11-error@-1 {{no matching constructor for initialization of 'B' (aka '(lambda at}}
  //   since-cxx11-note@#dr1891-b {{candidate constructor (the implicit copy constructor) not viable: requires 1 argument, but 0 were provided}}
  //   since-cxx11-note@#dr1891-b {{candidate constructor (the implicit move constructor) not viable: requires 1 argument, but 0 were provided}}

  // C++20 allows assignment for non-capturing lambdas (P0624R2).
  a = a;
  // cxx11-17-error-re@-1 {{{{object of type '\(lambda at .+\)' cannot be assigned because its copy assignment operator is implicitly deleted}}}}
  //   cxx11-17-note@#dr1891-a {{lambda expression begins here}}
  a = static_cast<A&&>(a);
  // cxx11-17-error-re@-1 {{{{object of type '\(lambda at .+\)' cannot be assigned because its copy assignment operator is implicitly deleted}}}}
  //   cxx11-17-note@#dr1891-a {{lambda expression begins here}}
  b = b;
  // since-cxx11-error-re@-1 {{{{object of type '\(lambda at .+\)' cannot be assigned because its copy assignment operator is implicitly deleted}}}}
  //   since-cxx11-note@#dr1891-b {{lambda expression begins here}}
  b = static_cast<B&&>(b);
  // since-cxx11-error-re@-1 {{{{object of type '\(lambda at .+\)' cannot be assigned because its copy assignment operator is implicitly deleted}}}}
  //   since-cxx11-note@#dr1891-b {{lambda expression begins here}}
#endif
}

namespace dr1894 { // dr1894: 3.8
                   // NB: reusing part of dr407 test
namespace A {
  struct S {};
}
namespace B {
  typedef int S;
}
namespace E {
  typedef A::S S;
  using A::S;
  struct S s;
}
namespace F {
  typedef A::S S;
}
namespace G {
  using namespace A;
  using namespace F;
  struct S s;
}
namespace H {
  using namespace F;
  using namespace A;
  struct S s;
}
}
