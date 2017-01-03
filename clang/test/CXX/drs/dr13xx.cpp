// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1z %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

namespace dr1315 { // dr1315: partial
  template <int I, int J> struct A {};
  template <int I> // expected-note {{non-deducible template parameter 'I'}}
    struct A<I + 5, I * 2> {}; // expected-error {{contains a template parameter that cannot be deduced}}
  template <int I> struct A<I, I> {};

  template <int I, int J, int K> struct B;
  template <int I, int K> struct B<I, I * 2, K> {}; // expected-note {{matches}}
  B<1, 2, 3> b1;

  // Multiple declarations with the same dependent expression are equivalent
  // for partial ordering purposes.
  template <int I> struct B<I, I * 2, 2> { typedef int type; };
  B<1, 2, 2>::type b2;

  // Multiple declarations with differing dependent expressions are unordered.
  template <int I, int K> struct B<I, I + 1, K> {}; // expected-note {{matches}}
  B<1, 2, 4> b3; // expected-error {{ambiguous}}

  // FIXME: Under dr1315, this is perhaps valid, but that is not clear: this
  // fails the "more specialized than the primary template" test because the
  // dependent type of T::value is not the same as 'int'.
  // A core issue will be opened to decide what is supposed to happen here.
  template <typename T, int I> struct C;
  template <typename T> struct C<T, T::value>;
  // expected-error@-1 {{type of specialized non-type template argument depends on a template parameter of the partial specialization}}
}

namespace dr1330 { // dr1330: 4.0 c++11
  // exception-specifications are parsed in a context where the class is complete.
  struct A {
    void f() throw(T) {} // expected-error 0-1{{C++1z}} expected-note 0-1{{noexcept}}
    struct T {};

#if __cplusplus >= 201103L
    void g() noexcept(&a == b) {}
    static int a;
    static constexpr int *b = &a;
#endif
  };

  void (A::*af1)() throw(A::T) = &A::f; // expected-error 0-1{{C++1z}} expected-note 0-1{{noexcept}}
  void (A::*af2)() throw() = &A::f; // expected-error-re {{{{not superset|different exception spec}}}}

#if __cplusplus >= 201103L
  static_assert(noexcept(A().g()), "");
#endif

  // Likewise, they're instantiated separately from an enclosing class template.
  template<typename U>
  struct B {
    void f() throw(T, typename U::type) {} // expected-error 0-1{{C++1z}} expected-note 0-1{{noexcept}}
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

  void (B<P>::*bpf1)() throw(B<P>::T, int) = &B<P>::f; // expected-error 0-1{{C++1z}} expected-note 0-1{{noexcept}}
#if __cplusplus < 201103L
  // expected-error@-2 {{not superset}}
  // FIXME: We only delay instantiation in C++11 onwards. In C++98, something
  // weird happens: instantiation of B<P> fails because it references T before
  // it's instantiated, but the diagnostic is suppressed in
  // Sema::FindInstantiatedDecl because we've already hit an error. This is
  // obviously a bad way to react to this situation; we should still producing
  // the "T has not yet been instantiated" error here, rather than giving
  // confusing errors later on.
#endif
  void (B<P>::*bpf2)() throw(int) = &B<P>::f; // expected-error 0-1{{C++1z}} expected-note 0-1{{noexcept}}
#if __cplusplus <= 201402L
  // expected-error@-2 {{not superset}}
#else
  // expected-warning@-4 {{not superset}}
#endif
  void (B<P>::*bpf3)() = &B<P>::f;
  void (B<P>::*bpf4)() throw() = &B<P>::f;
#if __cplusplus <= 201402L
  // expected-error@-2 {{not superset}}
#else
  // expected-error@-4 {{different exception specifications}}
#endif

#if __cplusplus >= 201103L
  static_assert(noexcept(B<P>().g()), "");
  struct Q { static const int value = false; };
  static_assert(!noexcept(B<Q>().g()), "");
#endif

  template<typename T> int f() throw(typename T::error) { return 0; } // expected-error 1-4{{prior to '::'}} expected-note 0-1{{instantiation of}}
#if __cplusplus > 201402L
    // expected-error@-2 0-1{{C++1z}} expected-note@-2 0-1{{noexcept}}
#endif
  // An exception-specification is needed even if the function is only used in
  // an unevaluated operand.
  int f1 = sizeof(f<int>()); // expected-note {{instantiation of}}
#if __cplusplus >= 201103L
  decltype(f<char>()) f2; // expected-note {{instantiation of}}
  bool f3 = noexcept(f<float>()); // expected-note {{instantiation of}}
#endif
  template int f<short>(); // expected-note {{instantiation of}}

  template<typename T> struct C {
    C() throw(typename T::type); // expected-error 1-2{{prior to '::'}}
#if __cplusplus > 201402L
    // expected-error@-2 0-1{{C++1z}} expected-note@-2 0-1{{noexcept}}
#endif
  };
  struct D : C<void> {}; // ok
#if __cplusplus < 201103L
  // expected-note@-2 {{instantiation of}}
#endif
  void f(D &d) { d = d; } // ok

  // FIXME: In C++11 onwards, we should also note the declaration of 'e' as the
  // line that triggers the use of E::E()'s exception specification.
  struct E : C<int> {}; // expected-note {{in instantiation of}}
  E e;
}

namespace dr1346 { // dr1346: 3.5
  auto a(1); // expected-error 0-1{{extension}}
  auto b(1, 2); // expected-error {{multiple expressions}} expected-error 0-1{{extension}}
#if __cplusplus >= 201103L
  auto c({}); // expected-error {{parenthesized initializer list}}
  auto d({1}); // expected-error {{parenthesized initializer list}}
  auto e({1, 2}); // expected-error {{parenthesized initializer list}}
#endif
  template<typename...Ts> void f(Ts ...ts) { // expected-error 0-1{{extension}}
    auto x(ts...); // expected-error {{empty}} expected-error 0-1{{extension}}
  }
  template void f(); // expected-note {{instantiation}}

#if __cplusplus >= 201103L
  void init_capture() {
    [a(1)] {} (); // expected-error 0-1{{extension}}
    [b(1, 2)] {} (); // expected-error {{multiple expressions}} expected-error 0-1{{extension}}
#if __cplusplus >= 201103L
    [c({})] {} (); // expected-error {{parenthesized initializer list}} expected-error 0-1{{extension}}
    [d({1})] {} (); // expected-error {{parenthesized initializer list}} expected-error 0-1{{extension}}
    [e({1, 2})] {} (); // expected-error {{parenthesized initializer list}} expected-error 0-1{{extension}}
#endif
  }
#endif
}

namespace dr1359 { // dr1359: 3.5
#if __cplusplus >= 201103L
  union A { constexpr A() = default; };
  union B { constexpr B() = default; int a; }; // expected-error {{not constexpr}} expected-note 2{{candidate}}
  union C { constexpr C() = default; int a, b; }; // expected-error {{not constexpr}} expected-note 2{{candidate}}
  struct X { constexpr X() = default; union {}; };
  struct Y { constexpr Y() = default; union { int a; }; }; // expected-error {{not constexpr}} expected-note 2{{candidate}}

  constexpr A a = A();
  constexpr B b = B(); // expected-error {{no matching}}
  constexpr C c = C(); // expected-error {{no matching}}
  constexpr X x = X();
  constexpr Y y = Y(); // expected-error {{no matching}}
#endif
}
