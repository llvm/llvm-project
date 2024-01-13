// RUN: %clang_cc1 -std=c++98 %s -verify=expected,cxx98-14,cxx98 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify=expected,cxx11-17,cxx11-14,cxx98-14,since-cxx11,cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify=expected,cxx11-17,cxx11-14,since-cxx14,cxx98-14,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 %s -verify=expected,cxx11-17,since-cxx14,since-cxx17,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++20 %s -verify=expected,since-cxx14,since-cxx20,since-cxx17,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++23 %s -verify=expected,since-cxx14,since-cxx20,since-cxx17,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++2c %s -verify=expected,since-cxx14,since-cxx20,since-cxx17,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors

__extension__ typedef __SIZE_TYPE__ size_t;

namespace std {
  template<typename T> struct initializer_list {
    const T *ptr;
    size_t n;
    initializer_list(const T*, size_t);
  };
}

#if __cplusplus >= 201103L
namespace dr1305 { // dr1305: 3.0
struct Incomplete; // #dr1305-Incomplete
struct Complete {};

int incomplete = alignof(Incomplete(&)[]);
// since-cxx11-error@-1 {{invalid application of 'alignof' to an incomplete type 'Incomplete'}}
//   since-cxx11-note@#dr1305-Incomplete {{forward declaration of 'dr1305::Incomplete'}}
int complete = alignof(Complete(&)[]);
}
#endif

namespace dr1307 { // dr1307: 14
#if __cplusplus >= 201103L
void f(int const (&)[2]);
void f(int const (&)[3]);

void caller() {
  // This should not be ambiguous, the 2nd overload is better.
  f({1, 2, 3});
}
#endif // __cplusplus >= 201103L
} // namespace dr1307

// dr1308: sup 1330

namespace dr1310 { // dr1310: 5
  struct S {} * sp = new S::S;
  // expected-error@-1 {{qualified reference to 'S' is a constructor name rather than a type in this context}}
  void f() {
    S::S(a);
    // expected-error@-1 {{qualified reference to 'S' is a constructor name rather than a type in this context}}
  }
  struct T { int n; typedef int U; typedef T V; };
  int k = T().T::T::n;
  T::V v;

  struct U { int U; };
  int u = U().U::U;
  struct U::U w;

  struct V : T::T {
    // FIXME: This is technically ill-formed, but we consider that to be a defect.
    V() : T::T() {}
  };
  template<typename T> struct VT : T::T {
    VT() : T::T() {}
  };
  template struct VT<T>;

  template<template<typename> class> class TT {};
  template<typename> class TTy {};

  template<typename T> struct WBase {};
  template<typename T> struct W : WBase<T> { typedef int X; int n; };

  void w_test() {
    W<int>::W w1a;
    // expected-error@-1 {{qualified reference to 'W' is a constructor name rather than a type in this context}}
    W<int>::W::X w1ax;
    W<int>::W<int> w1b;
    // expected-error@-1 {{qualified reference to 'W' is a constructor name rather than a template name in this context}}
    W<int>::W<int>::X w1bx;
    typename W<int>::W w2a;
    // expected-error@-1 {{ISO C++ specifies that qualified reference to 'W' is a constructor name rather than a type in this context, despite preceding 'typename' keyword}}
    // cxx98-error@-2 {{'typename' occurs outside of a template}}
    typename W<int>::W::X w2ax;
    // cxx98-error@-1 {{'typename' occurs outside of a template}}
    typename W<int>::W<int> w2b;
    // expected-error@-1 {{ISO C++ specifies that qualified reference to 'W' is a constructor name rather than a template name in this context, despite preceding 'typename' keyword}}
    // cxx98-error@-2 {{'typename' occurs outside of a template}}
    typename W<int>::W<int>::X w2bx;
    // cxx98-error@-1 {{'typename' occurs outside of a template}}
    W<int>::template W<int> w3;
    // expected-error@-1 {{ISO C++ specifies that qualified reference to 'W' is a constructor name rather than a template name in this context, despite preceding 'template' keyword}}
    // cxx98-error@-2 {{'template' keyword outside of a template}}
    W<int>::template W<int>::X w3x;
    // cxx98-error@-1 {{'template' keyword outside of a template}}
    typename W<int>::template W<int> w4;
    // expected-error@-1 {{ISO C++ specifies that qualified reference to 'W' is a constructor name rather than a template name in this context, despite preceding 'template' keyword}}
    // cxx98-error@-2 {{'template' keyword outside of a template}}
    // cxx98-error@-3 {{'typename' occurs outside of a template}}
    typename W<int>::template W<int>::X w4x;
    // cxx98-error@-1 {{'template' keyword outside of a template}}
    // cxx98-error@-2 {{'typename' occurs outside of a template}}

    TT<W<int>::W> tt1;
    // expected-error@-1 {{qualified reference to 'W' is a constructor name rather than a type in this context}}
    TTy<W<int>::W> tt1a;
    // expected-error@-1 {{qualified reference to 'W' is a constructor name rather than a type in this context}}
    TT<W<int>::template W> tt2;
    // expected-error@-1 {{ISO C++ specifies that qualified reference to 'W' is a constructor name rather than a template name in this context, despite preceding 'template' keyword}}
    // cxx98-error@-2 {{'template' keyword outside of a template}}
    TT<W<int>::WBase> tt3;
    TTy<W<int>::WBase> tt3a;
    TT<W<int>::template WBase> tt4;
    // cxx98-error@-1 {{'template' keyword outside of a template}}

    W<int> w;
    (void)w.W::W::n;
    (void)w.W<int>::W::n;
    (void)w.W<int>::W<int>::n;
    (void)w.W<int>::template W<int>::n;
    // cxx98-error@-1 {{'template' keyword outside of a template}}
  }

  template<typename W>
  void wt_test() {
    typename W::W w2a;
    // expected-error@-1 {{ISO C++ specifies that qualified reference to 'W' is a constructor name rather than a type in this context, despite preceding 'typename' keyword}}
    //   cxx98-note@#dr1310-W-int {{in instantiation of function template specialization 'dr1310::wt_test<dr1310::W<int> >' requested here}}
    //   since-cxx11-note@#dr1310-W-int {{in instantiation of function template specialization 'dr1310::wt_test<dr1310::W<int>>' requested here}}
    typename W::template W<int> w4;
    // expected-error@-1 {{ISO C++ specifies that qualified reference to 'W' is a constructor name rather than a template name in this context, despite preceding 'template' keyword}}
    TTy<typename W::W> tt2;
    // expected-error@-1 {{ISO C++ specifies that qualified reference to 'W' is a constructor name rather than a type in this context, despite preceding 'typename' keyword}}
    TT<W::template W> tt3;
    // expected-error@-1 {{ISO C++ specifies that qualified reference to 'W' is a constructor name rather than a template name in this context, despite preceding 'template' keyword}}
  }
  template<typename W>
  void wt_test_good() {
    typename W::W::X w2ax;
    typename W::template W<int>::X w4x;
    TTy<typename W::WBase> tt4;
    TT<W::template WBase> tt5;

    W w;
    (void)w.W::W::n;
    (void)w.W::template W<int>::n;
    (void)w.template W<int>::W::n;
    (void)w.template W<int>::template W<int>::n;
  }
  template void wt_test<W<int> >(); // #dr1310-W-int
  template void wt_test_good<W<int> >();
}

namespace dr1315 { // dr1315: partial
  template <int I, int J> struct A {};
  template <int I> struct A<I + 5, I * 2> {};
  // expected-error@-1 {{class template partial specialization contains a template parameter that cannot be deduced; this partial specialization will never be used}}
  //   expected-note@-2 {{non-deducible template parameter 'I'}}
  template <int I> struct A<I, I> {};

  template <int I, int J, int K> struct B;
  template <int I, int K> struct B<I, I * 2, K> {}; // #dr1315-B-1
  B<1, 2, 3> b1;

  // Multiple declarations with the same dependent expression are equivalent
  // for partial ordering purposes.
  template <int I> struct B<I, I * 2, 2> { typedef int type; };
  B<1, 2, 2>::type b2;

  // Multiple declarations with differing dependent expressions are unordered.
  template <int I, int K> struct B<I, I + 1, K> {}; // #dr1315-B-2
  B<1, 2, 4> b3;
  // expected-error@-1 {{ambiguous partial specializations of 'B<1, 2, 4>'}}
  //   expected-note@#dr1315-B-1 {{partial specialization matches [with I = 1, K = 4]}}
  //   expected-note@#dr1315-B-2 {{partial specialization matches [with I = 1, K = 4]}}

  // FIXME: Under dr1315, this is perhaps valid, but that is not clear: this
  // fails the "more specialized than the primary template" test because the
  // dependent type of T::value is not the same as 'int'.
  // A core issue will be opened to decide what is supposed to happen here.
  template <typename T, int I> struct C;
  template <typename T> struct C<T, T::value>;
  // expected-error@-1 {{type of specialized non-type template argument depends on a template parameter of the partial specialization}}
}

namespace dr1330 { // dr1330: 4 c++11
  // exception-specifications are parsed in a context where the class is complete.
  struct A {
    void f() throw(T) {}
    // since-cxx17-error@-1 {{ISO C++17 does not allow dynamic exception specifications}}
    //   since-cxx17-note@-2 {{use 'noexcept(false)' instead}}
    struct T {};

#if __cplusplus >= 201103L
    void g() noexcept(&a == b) {}
    static int a;
    static constexpr int *b = &a;
#endif
  };

  void (A::*af1)() throw(A::T) = &A::f;
  // since-cxx17-error@-1 {{ISO C++17 does not allow dynamic exception specifications}}
  //   since-cxx17-note@-2 {{use 'noexcept(false)' instead}}
  void (A::*af2)() throw() = &A::f;
  // cxx98-14-error@-1 {{target exception specification is not superset of source}}
  // since-cxx17-error@-2 {{cannot initialize a variable of type 'void (dr1330::A::*)() throw()' with an rvalue of type 'void (dr1330::A::*)() throw(T)': different exception specifications}}

#if __cplusplus >= 201103L
  static_assert(noexcept(A().g()), "");
#endif

  // Likewise, they're instantiated separately from an enclosing class template.
  template<typename U>
  struct B {
    void f() throw(T, typename U::type) {}
    // since-cxx17-error@-1 {{ISO C++17 does not allow dynamic exception specifications}}
    //   since-cxx17-note@-2 {{use 'noexcept(false)' instead}}
    struct T {};

#if __cplusplus >= 201103L
    void g() noexcept(&a == b && U::value) {}
    static int a;
    static constexpr int *b = &a;
#endif
  };

  B<int> bi; // ok

  struct P {
    typedef int type;
    static const int value = true;
  };

  // FIXME: We only delay instantiation in C++11 onwards. In C++98, something
  // weird happens: instantiation of B<P> fails because it references T before
  // it's instantiated, but the diagnostic is suppressed in
  // Sema::FindInstantiatedDecl because we've already hit an error. This is
  // obviously a bad way to react to this situation; we should still producing
  // the "T has not yet been instantiated" error here, rather than giving
  // confusing errors later on.
  void (B<P>::*bpf1)() throw(B<P>::T, int) = &B<P>::f;
  // since-cxx17-error@-1 {{ISO C++17 does not allow dynamic exception specifications}}
  //   since-cxx17-note@-2 {{use 'noexcept(false)' instead}}
  // cxx98-error@-3 {{target exception specification is not superset of source}}

  void (B<P>::*bpf2)() throw(int) = &B<P>::f;
  // since-cxx17-error@-1 {{ISO C++17 does not allow dynamic exception specifications}}
  //   since-cxx17-note@-2 {{use 'noexcept(false)' instead}}
  // cxx98-14-error@-3 {{target exception specification is not superset of source}}
  // since-cxx17-warning@-4 {{target exception specification is not superset of source}}
  void (B<P>::*bpf3)() = &B<P>::f;
  void (B<P>::*bpf4)() throw() = &B<P>::f;
  // cxx98-14-error@-1 {{target exception specification is not superset of source}}
  // since-cxx17-error@-2 {{cannot initialize a variable of type 'void (B<P>::*)() throw()' with an rvalue of type 'void (dr1330::B<dr1330::P>::*)() throw(T, typename P::type)': different exception specifications}}

#if __cplusplus >= 201103L
  static_assert(noexcept(B<P>().g()), "");
  struct Q { static const int value = false; };
  static_assert(!noexcept(B<Q>().g()), "");
#endif

  template<typename T> int f() throw(typename T::error) { return 0; } // #dr1330-f
  // expected-error@#dr1330-f {{type 'int' cannot be used prior to '::' because it has no members}}
  //   cxx98-note@#dr1330-f-int {{in instantiation of function template specialization 'dr1330::f<int>' requested here}}
  //   since-cxx11-note@#dr1330-f-int {{in instantiation of exception specification for 'f<int>' requested here}}
  // cxx98-14-error@#dr1330-f {{type 'short' cannot be used prior to '::' because it has no members}}
  //   cxx98-14-note@#dr1330-f-short {{in instantiation of function template specialization 'dr1330::f<short>' requested here}}
  //   cxx11-14-note@#dr1330-f {{in instantiation of exception specification for 'f<short>' requested here}}
  // since-cxx11-error@#dr1330-f {{type 'char' cannot be used prior to '::' because it has no members}}
  //   since-cxx11-note@#dr1330-f-char {{in instantiation of exception specification for 'f<char>' requested here}}
  // since-cxx11-error@#dr1330-f {{type 'float' cannot be used prior to '::' because it has no members}}
  //   since-cxx11-note@#dr1330-f-float {{in instantiation of exception specification for 'f<float>' requested here}}
  // since-cxx17-error@#dr1330-f {{ISO C++17 does not allow dynamic exception specifications}}
  //   since-cxx17-note@#dr1330-f {{use 'noexcept(false)' instead}}

  // An exception-specification is needed even if the function is only used in
  // an unevaluated operand.
  int f1 = sizeof(f<int>()); // #dr1330-f-int
#if __cplusplus >= 201103L
  decltype(f<char>()) f2; // #dr1330-f-char
  bool f3 = noexcept(f<float>()); /// #dr1330-f-float
#endif
  // In C++17 onwards, substituting explicit template arguments into the
  // function type substitutes into the exception specification (because it's
  // part of the type). In earlier languages, we don't notice there's a problem
  // until we've already started to instantiate.
  template int f<short>(); // #dr1330-f-short
  // since-cxx17-error@-1 {{explicit instantiation of 'f' does not refer to a function template, variable template, member function, member class, or static data member}}
  //   since-cxx17-note@#dr1330-f {{candidate template ignored: substitution failure [with T = short]: type 'short' cannot be used prior to '::' because it has no members}}

  template<typename T> struct C {
    C() throw(typename T::type); // #dr1330-C
    // since-cxx17-error@-1 {{ISO C++17 does not allow dynamic exception specifications}}
    //   since-cxx17-note@-2 {{use 'noexcept(false)' instead}}
    // cxx98-error@#dr1330-C {{type 'void' cannot be used prior to '::' because it has no members}}
    //   cxx98-note@#dr1330-C-void {{in instantiation of template class 'dr1330::C<void>' requested here}}
    // expected-error@#dr1330-C {{type 'int' cannot be used prior to '::' because it has no members}}
    //   cxx98-note@#dr1330-C-int {{in instantiation of template class 'dr1330::C<int>' requested here}}
    //   since-cxx11-note@#dr1330-C-int {{in instantiation of exception specification for 'C' requested here}}
    //   since-cxx11-note@#dr1330-e {{in evaluation of exception specification for 'dr1330::E::E' needed here}}
  };
  struct D : C<void> {}; // #dr1330-C-void
  void f(D &d) { d = d; } // ok

  struct E : C<int> {}; // #dr1330-C-int
  E e; // #dr1330-e
}

namespace dr1341 { // dr1341: sup P0683R1
#if __cplusplus >= 202002L
int a;
const int b = 0; // #dr1341-b
struct S {
  int x1 : 8 = 42;
  int x2 : 8 { 42 };
  int y1 : true ? 8 : a = 42;
  int y2 : true ? 8 : b = 42;
  // since-cxx20-error@-1 {{cannot assign to variable 'b' with const-qualified type 'const int'}}
  //   since-cxx20-note@#dr1341-b {{variable 'b' declared const here}}
  int y3 : (true ? 8 : b) = 42;
  int z : 1 || new int { 0 };
};
#endif
}

namespace dr1346 { // dr1346: 3.5
  auto a(1);
  // cxx98-error@-1 {{'auto' type specifier is a C++11 extension}}
  auto b(1, 2);
  // cxx98-error@-1 {{'auto' type specifier is a C++11 extension}}
  // expected-error@-2 {{initializer for variable 'b' with type 'auto' contains multiple expressions}}
#if __cplusplus >= 201103L
  auto c({});
  // since-cxx11-error@-1 {{cannot deduce type for variable 'c' with type 'auto' from parenthesized initializer list}}
  auto d({1});
  // since-cxx11-error@-1 {{cannot deduce type for variable 'd' with type 'auto' from parenthesized initializer list}}
  auto e({1, 2});
  // since-cxx11-error@-1 {{cannot deduce type for variable 'e' with type 'auto' from parenthesized initializer list}}
#endif
  template<typename...Ts> void f(Ts ...ts) {
  // cxx98-error@-1 {{variadic templates are a C++11 extension}}
    auto x(ts...);
    // cxx98-error@-1 {{'auto' type specifier is a C++11 extension}}
    // expected-error@-2 {{initializer for variable 'x' with type 'auto' is empty}}
    //   expected-note@#dr1346-f {{in instantiation of function template specialization 'dr1346::f<>' requested here}}
  }
  template void f(); // #dr1346-f

#if __cplusplus >= 201103L
  void init_capture() {
    [a(1)] {} ();
    // cxx11-error@-1 {{initialized lambda captures are a C++14 extension}}
    [b(1, 2)] {} ();
    // cxx11-error@-1 {{initialized lambda captures are a C++14 extension}}
    // since-cxx11-error@-2 {{initializer for lambda capture 'b' contains multiple expressions}}
    [c({})] {} ();
    // cxx11-error@-1 {{initialized lambda captures are a C++14 extension}}
    // since-cxx11-error@-2 {{cannot deduce type for lambda capture 'c' from parenthesized initializer list}}
    [d({1})] {} ();
    // cxx11-error@-1 {{initialized lambda captures are a C++14 extension}}
    // since-cxx11-error@-2 {{cannot deduce type for lambda capture 'd' from parenthesized initializer list}}
    [e({1, 2})] {} ();
    // cxx11-error@-1 {{initialized lambda captures are a C++14 extension}}
    // since-cxx11-error@-2 {{cannot deduce type for lambda capture 'e' from parenthesized initializer list}}
  }
#endif
}

namespace dr1347 { // dr1347: 3.1
  auto x = 5, *y = &x;
  // cxx98-error@-1 {{'auto' type specifier is a C++11 extension}}
  auto z = y, *q = y;
  // cxx98-error@-1 {{'auto' type specifier is a C++11 extension}}
  // expected-error@-2 {{'auto' deduced as 'int *' in declaration of 'z' and deduced as 'int' in declaration of 'q'}}
#if __cplusplus >= 201103L
  auto a = 5, b = {1, 2};
  // since-cxx11-error@-1 {{'auto' deduced as 'int' in declaration of 'a' and deduced as 'std::initializer_list<int>' in declaration of 'b'}}
  auto (*fp)(int) -> int, i = 0;
  // since-cxx11-error@-1 {{declaration with trailing return type must be the only declaration in its group}}
#endif
}

namespace dr1350 { // dr1350: 3.5
#if __cplusplus >= 201103L
struct NoexceptCtor {
  NoexceptCtor(int) noexcept {}
};

struct ThrowingNSDMI : NoexceptCtor {
  int a = []() noexcept(false) { return 0; }();
  using NoexceptCtor::NoexceptCtor;
};

static_assert(!__is_nothrow_constructible(ThrowingNSDMI, int), "");

struct ThrowingCtor {
  ThrowingCtor() noexcept(false) {}
};

struct ThrowingNSDM : NoexceptCtor {
  ThrowingCtor c;
  using NoexceptCtor::NoexceptCtor;
};

static_assert(!__is_nothrow_constructible(ThrowingNSDM, int), "");

struct ThrowingCtorTemplate {
  template <typename = int>
  ThrowingCtorTemplate() noexcept(false) {}
};

struct ThrowingNSDM2 : NoexceptCtor {
  ThrowingCtorTemplate c;
  using NoexceptCtor::NoexceptCtor;
};

static_assert(!__is_nothrow_constructible(ThrowingNSDM2, int), "");

struct D1 : NoexceptCtor, ThrowingCtor {
  using NoexceptCtor::NoexceptCtor;
};

static_assert(!__is_nothrow_constructible(D1, int), "");

struct D2 : NoexceptCtor, ThrowingCtorTemplate {
  using NoexceptCtor::NoexceptCtor;
};

static_assert(!__is_nothrow_constructible(D2, int), "");

struct ThrowingDefaultArg {
  ThrowingDefaultArg(ThrowingCtor = {}) {}
};

struct D3 : NoexceptCtor, ThrowingDefaultArg {
  using NoexceptCtor::NoexceptCtor;
};

static_assert(!__is_nothrow_constructible(D3, int), "");

struct ThrowingDefaultArgTemplate {
  template <typename = int>
  ThrowingDefaultArgTemplate(ThrowingCtor = {}) {}
};

struct D4 : NoexceptCtor, ThrowingDefaultArgTemplate {
  using NoexceptCtor::NoexceptCtor;
};

static_assert(!__is_nothrow_constructible(D4, int), "");
#endif
} // namespace dr1350

namespace dr1358 { // dr1358: 3.1
#if __cplusplus >= 201103L
  struct Lit { constexpr operator int() const { return 0; } };
  struct NonLit { NonLit(); operator int(); }; // #dr1358-NonLit
  struct NonConstexprConv { constexpr operator int() const; };
  struct Virt { virtual int f(int) const; };

  template<typename T, typename U, typename V> struct A : V {
    int member;
    constexpr A(U u) : member(u) {}
    constexpr T f(U u) const { return T(); }
  };

  constexpr A<Lit, Lit, Lit> ce = Lit();
  constexpr int k = ce.f(Lit{});

  // Can have a non-literal return type and parameter type.
  // Constexpr function can be implicitly virtual.
  A<NonLit, NonLit, Virt> a = NonLit();
  void g() { a.f(NonLit()); }

  // Constructor is still constexpr, so this is a literal type.
  static_assert(__is_literal_type(decltype(a)), "");

  // Constructor can call non-constexpr functions.
  A<Lit, NonConstexprConv, Lit> b = NonConstexprConv();

  // But the corresponding non-template cases are rejected.
  struct B : Virt {
    int member;
    constexpr B(NonLit u) : member(u) {}
    // since-cxx11-error@-1 {{constexpr constructor's 1st parameter type 'NonLit' is not a literal type}}
    //   since-cxx11-note@#dr1358-NonLit {{'NonLit' is not literal because it is not an aggregate and has no constexpr constructors other than copy or move constructors}}
    constexpr NonLit f(NonLit u) const { return NonLit(); }
    // since-cxx11-error@-1 {{constexpr function's return type 'NonLit' is not a literal type}}
    //   since-cxx11-note@#dr1358-NonLit {{'NonLit' is not literal because it is not an aggregate and has no constexpr constructors other than copy or move constructors}}
  };
#endif
}

namespace dr1359 { // dr1359: 3.5
#if __cplusplus >= 201103L
  union A { constexpr A() = default; };
  union B { constexpr B() = default; int a; }; // #dr1359-B
  // cxx11-17-error@-1 {{defaulted definition of default constructor is not constexpr}}
  union C { constexpr C() = default; int a, b; }; // #dr1359-C
  // cxx11-17-error@-1 {{defaulted definition of default constructor is not constexpr}} 
  struct X { constexpr X() = default; union {}; };
  // since-cxx11-error@-1 {{declaration does not declare anything}}
  struct Y { constexpr Y() = default; union { int a; }; }; // #dr1359-Y
  // cxx11-17-error@-1 {{defaulted definition of default constructor is not constexpr}}

  constexpr A a = A();
  constexpr B b = B();
  // cxx11-17-error@-1 {{no matching constructor for initialization of 'B'}}
  //   cxx11-17-note@#dr1359-B {{candidate constructor (the implicit copy constructor) not viable: requires 1 argument, but 0 were provided}}
  //   cxx11-17-note@#dr1359-B {{candidate constructor (the implicit move constructor) not viable: requires 1 argument, but 0 were provided}}
  constexpr C c = C();
  // cxx11-17-error@-1 {{no matching constructor for initialization of 'C'}}
  //   cxx11-17-note@#dr1359-C {{candidate constructor (the implicit copy constructor) not viable: requires 1 argument, but 0 were provided}}
  //   cxx11-17-note@#dr1359-C {{candidate constructor (the implicit move constructor) not viable: requires 1 argument, but 0 were provided}}
  constexpr X x = X();
  constexpr Y y = Y();
  // cxx11-17-error@-1 {{no matching constructor for initialization of 'Y'}}
  //   cxx11-17-note@#dr1359-Y {{candidate constructor (the implicit copy constructor) not viable: requires 1 argument, but 0 were provided}}
  //   cxx11-17-note@#dr1359-Y {{candidate constructor (the implicit move constructor) not viable: requires 1 argument, but 0 were provided}}
#endif
}

namespace dr1388 { // dr1388: 4
  template<typename A, typename ...T> void f(T..., A); // #dr1388-f
  // cxx98-error@-1 {{variadic templates are a C++11 extension}}
  template<typename ...T> void g(T..., int); // #dr1388-g
  // cxx98-error@-1 {{variadic templates are a C++11 extension}}
  template<typename ...T, typename A> void h(T..., A); // #dr1388-h
  // cxx98-error@-1 {{variadic templates are a C++11 extension}}

  void test_f() {
    f(0); // ok, trailing parameter pack deduced to empty
    f(0, 0);
    // expected-error@-1 {{no matching function for call to 'f'}}
    //   expected-note@#dr1388-f {{candidate function [with A = int, T = <>] not viable: requires 1 argument, but 2 were provided}}
    f<int>(0);
    f<int>(0, 0);
    // expected-error@-1 {{no matching function for call to 'f'}}
    //   expected-note@#dr1388-f {{candidate function [with A = int, T = <>] not viable: requires 1 argument, but 2 were provided}}
    f<int, int>(0, 0);
    f<int, int, int>(0, 0);
    // expected-error@-1 {{no matching function for call to 'f'}}
    //   expected-note@#dr1388-f {{candidate function [with A = int, T = <int, int>] not viable: requires 3 arguments, but 2 were provided}}

    g(0);
    g(0, 0);
    // expected-error@-1 {{no matching function for call to 'g'}}
    //   expected-note@#dr1388-g {{candidate function [with T = <>] not viable: requires 1 argument, but 2 were provided}}
    g<>(0);
    g<int>(0);
    // expected-error@-1 {{no matching function for call to 'g'}}
    //   expected-note@#dr1388-g {{candidate function [with T = <int>] not viable: requires 2 arguments, but 1 was provided}}
    g<int>(0, 0);

    h(0);
    h(0, 0);
    // expected-error@-1 {{no matching function for call to 'h'}}
    //   expected-note@#dr1388-h {{candidate function [with T = <>, A = int] not viable: requires 1 argument, but 2 were provided}}
    h<int>(0, 0);
    h<int, int>(0, 0);
    // expected-error@-1 {{no matching function for call to 'h'}}
    //   expected-note@#dr1388-h {{candidate template ignored: couldn't infer template argument 'A'}}
  }

  // A non-trailing parameter pack is still a non-deduced context, even though
  // we know exactly how many arguments correspond to it.
  template<typename T, typename U> struct pair {};
  template<typename ...T> struct tuple { typedef char type; }; //
  // cxx98-error@-1 {{variadic templates are a C++11 extension}}
  template<typename ...T, typename ...U> void f_pair_1(pair<T, U>..., int); // #dr1388-f-1
  // cxx98-error@-1 {{variadic templates are a C++11 extension}}
  // cxx98-error@-2 {{variadic templates are a C++11 extension}}
  template<typename ...T, typename U> void f_pair_2(pair<T, char>..., U);
  // cxx98-error@-1 {{variadic templates are a C++11 extension}}
  template<typename ...T, typename ...U> void f_pair_3(pair<T, U>..., tuple<U...>); // #dr1388-f-3
  // cxx98-error@-1 {{variadic templates are a C++11 extension}}
  // cxx98-error@-2 {{variadic templates are a C++11 extension}}
  template<typename ...T> void f_pair_4(pair<T, char>..., T...); // #dr1388-f-4
  // cxx98-error@-1 {{variadic templates are a C++11 extension}}
  void g(pair<int, char> a, pair<long, char> b, tuple<char, char> c) {
    f_pair_1<int, long>(a, b, 0);
    // expected-error@-1 {{no matching function for call to 'f_pair_1'}}
    //   expected-note@#dr1388-f-1 {{candidate template ignored: substitution failure [with T = <int, long>]: deduced incomplete pack <(no value), (no value)> for template parameter 'U'}}
    f_pair_2<int, long>(a, b, 0);
    f_pair_3<int, long>(a, b, c);
    f_pair_3<int, long>(a, b, tuple<char>());
    // expected-error@-1 {{no matching function for call to 'f_pair_3'}}
    //   expected-note@#dr1388-f-3 {{candidate template ignored: deduced packs of different lengths for parameter 'U' (<(no value), (no value)> vs. <char>)}}
    f_pair_4<int, long>(a, b, 0, 0L);
    f_pair_4<int, long>(a, b, 0, 0L, "foo");
    // expected-error@-1 {{no matching function for call to 'f_pair_4'}}
    //   expected-note@#dr1388-f-4 {{candidate template ignored: deduced packs of different lengths for parameter 'T' (<int, long> vs. <int, long, const char *>)}}
  }
}

namespace dr1391 { // dr1391: partial
  struct A {}; struct B : A {};
  template<typename T> struct C { C(int); typename T::error error; }; // #dr1391-C
  // expected-error@#dr1391-C {{type 'int' cannot be used prior to '::' because it has no members}}
  //   expected-note@#dr1391-b {{in instantiation of template class 'dr1391::C<int>' requested here}}
  //   expected-note@#dr1391-b {{while substituting deduced template arguments into function template 'b' [with T = int]}}
  // expected-error@#dr1391-C {{type 'double' cannot be used prior to '::' because it has no members}}
  //   expected-note@#dr1391-c {{in instantiation of template class 'dr1391::C<double>' requested here}}
  template<typename T> struct D {};

  // No deduction is performed for parameters with no deducible template-parameters, therefore types do not need to match.
  template<typename T> void a(T, int T::*);
  void test_a(int A::*p) { a(A(), p); } // ok, type of second parameter does not need to match

  namespace dr_example_1 {
    template<typename T, typename U> void f(C<T>);
    template<typename T> void f(D<T>);

    void g(D<int> d) {
      f(d); // ok, first 'f' eliminated by deduction failure
      f<int>(d); // ok, first 'f' eliminated because 'U' cannot be deduced
    }
  }

  namespace dr_example_2 {
    template<typename T> typename C<T>::error f(int, T);
    template<typename T> T f(T, T);

    void g(A a) {
      f(a, a); // ok, no conversion from A to int for first parameter of first candidate
    }
  }

  namespace std_example {
    template<typename T> struct Z {
      typedef typename T::x xx;
    };
    template<typename T> typename Z<T>::xx f(void *, T);
    template<typename T> void f(int, T);
    struct A {} a;
    void g() { f(1, a); }
  }

  template<typename T> void b(C<int> ci, T *p);
  void b(...);
  void test_b() {
    b(0, 0); // ok, deduction fails prior to forming a conversion sequence and instantiating C<int>
    // FIXME: The "while substituting" note should point at the overload candidate.
    b<int>(0, 0); // #dr1391-b
  }

  template<typename T> struct Id { typedef T type; };
  template<typename T> void c(T, typename Id<C<T> >::type);
  void test_c() {
    // Implicit conversion sequences for dependent types are checked later.
    c(0.0, 0); // #dr1391-c
  }

  namespace partial_ordering {
    // FIXME: Second template should be considered more specialized because non-dependent parameter is ignored.
    template<typename T> int a(T, short) = delete; // #dr1391-a-short
    // cxx98-error@-1 {{deleted function definitions are a C++11 extension}}
    template<typename T> int a(T*, char); // #dr1391-a-char
    int test_a = a((int*)0, 0);
    // expected-error@-1 {{call to 'a' is ambiguous}} FIXME
    //   expected-note@#dr1391-a-short {{candidate function [with T = int *] has been explicitly deleted}}
    //   expected-note@#dr1391-a-char {{candidate function [with T = int]}}

    // FIXME: Second template should be considered more specialized:
    // deducing #1 from #2 ignores the second P/A pair, so deduction succeeds,
    // deducing #2 from #1 fails to deduce T, so deduction fails.
    template<typename T> int b(T, int) = delete; // #dr1391-b-int
    // cxx98-error@-1 {{deleted function definitions are a C++11 extension}}
    template<typename T, typename U> int b(T*, U); // #dr1391-b-U
    int test_b = b((int*)0, 0);
    // expected-error@-1 {{call to 'b' is ambiguous}} FIXME
    //   expected-note@#dr1391-b-int {{candidate function [with T = int *] has been explicitly deleted}}
    //   expected-note@#dr1391-b-U {{candidate function [with T = int, U = int]}}

    // Unintended consequences: because partial ordering does not consider
    // explicit template arguments, and deduction from a non-dependent type
    // vacuously succeeds, a non-dependent template is less specialized than
    // anything else!
    // According to DR1391, this is ambiguous!
    template<typename T> int c(int);
    template<typename T> int c(T);
    int test_c1 = c(0); // ok
    int test_c2 = c<int>(0); // FIXME: apparently ambiguous
  }
}

namespace dr1394 { // dr1394: 15
#if __cplusplus >= 201103L
struct Incomplete;
Incomplete f(Incomplete) = delete; // well-formed
#endif
}

namespace dr1395 { // dr1395: 16
#if __cplusplus >= 201103L
  template <typename T, typename... U> void f(T, U...);
  template <typename T> void f(T);
  void h(int i) {
    // This is made ambiguous by dr692, but made valid again by dr1395.
    f(&i);
  }
#endif
}

namespace dr1397 { // dr1397: 3.2
#if __cplusplus >= 201103L
struct A {
// cxx11-error@-1 {{default member initializer for 'p' needed within definition of enclosing class 'A' outside of member functions}}
//   cxx11-note@#dr1397-p {{in evaluation of exception specification for 'dr1397::A::A' needed here}}
//   cxx11-note@#dr1397-p {{default member initializer declared here}}
  void *p = A{}; // #dr1397-p
  // since-cxx14-error@-1 {{default member initializer for 'p' needed within definition of enclosing class 'A' outside of member functions}}
  //   since-cxx14-note@-2 {{default member initializer declared here}}
  operator void*() const { return nullptr; }
};
#endif
} // namespace dr1397

namespace dr1399 { // dr1399: dup 1388
  template<typename ...T> void f(T..., int, T...) {} // #dr1399-f 
  // cxx98-error@-1 {{variadic templates are a C++11 extension}}
  void g() {
    f(0);
    f<int>(0, 0, 0);
    f(0, 0, 0);
    // expected-error@-1 {{no matching function for call to 'f'}}
    //   expected-note@#dr1399-f {{candidate template ignored: deduced packs of different lengths for parameter 'T' (<> vs. <int, int>)}}
  }
}
