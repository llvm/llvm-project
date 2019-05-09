// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

__extension__ typedef __SIZE_TYPE__ size_t;

namespace std {
  template<typename T> struct initializer_list {
    const T *ptr;
    size_t n;
    initializer_list(const T*, size_t);
  };
}

namespace dr1310 { // dr1310: 5
  struct S {} * sp = new S::S; // expected-error {{qualified reference to 'S' is a constructor name}}
  void f() {
    S::S(a); // expected-error {{qualified reference to 'S' is a constructor name}}
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
    W<int>::W w1a; // expected-error {{qualified reference to 'W' is a constructor name}}
    W<int>::W::X w1ax;
    W<int>::W<int> w1b; // expected-error {{qualified reference to 'W' is a constructor name}}
    W<int>::W<int>::X w1bx;
    typename W<int>::W w2a; // expected-error {{qualified reference to 'W' is a constructor name}} expected-error 0-1{{outside of a template}}
    typename W<int>::W::X w2ax; // expected-error 0-1{{outside of a template}}
    typename W<int>::W<int> w2b; // expected-error {{qualified reference to 'W' is a constructor name}} expected-error 0-1{{outside of a template}}
    typename W<int>::W<int>::X w2bx; // expected-error 0-1{{outside of a template}}
    W<int>::template W<int> w3; // expected-error {{qualified reference to 'W' is a constructor name}} expected-error 0-1{{outside of a template}}
    W<int>::template W<int>::X w3x; // expected-error 0-1{{outside of a template}}
    typename W<int>::template W<int> w4; // expected-error {{qualified reference to 'W' is a constructor name}} expected-error 0-2{{outside of a template}}
    typename W<int>::template W<int>::X w4x; // expected-error 0-2{{outside of a template}}

    TT<W<int>::W> tt1; // expected-error {{qualified reference to 'W' is a constructor name}}
    TTy<W<int>::W> tt1a; // expected-error {{qualified reference to 'W' is a constructor name}}
    TT<W<int>::template W> tt2; // expected-error {{qualified reference to 'W' is a constructor name}} expected-error 0-1{{outside of a template}}
    TT<W<int>::WBase> tt3;
    TTy<W<int>::WBase> tt3a;
    TT<W<int>::template WBase> tt4; // expected-error 0-1{{outside of a template}}

    W<int> w;
    (void)w.W::W::n;
    (void)w.W<int>::W::n;
    (void)w.W<int>::W<int>::n;
    (void)w.W<int>::template W<int>::n; // expected-error 0-1{{outside of a template}}
  }

  template<typename W>
  void wt_test() {
    typename W::W w2a; // expected-error {{qualified reference to 'W' is a constructor name}}
    typename W::template W<int> w4; // expected-error {{qualified reference to 'W' is a constructor name}}
    TTy<typename W::W> tt2; // expected-error {{qualified reference to 'W' is a constructor name}}
    TT<W::template W> tt3; // expected-error {{qualified reference to 'W' is a constructor name}}
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
  template void wt_test<W<int> >(); // expected-note {{instantiation of}}
  template void wt_test_good<W<int> >();
}

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

namespace dr1330 { // dr1330: 4 c++11
  // exception-specifications are parsed in a context where the class is complete.
  struct A {
    void f() throw(T) {} // expected-error 0-1{{C++17}} expected-note 0-1{{noexcept}}
    struct T {};

#if __cplusplus >= 201103L
    void g() noexcept(&a == b) {}
    static int a;
    static constexpr int *b = &a;
#endif
  };

  void (A::*af1)() throw(A::T) = &A::f; // expected-error 0-1{{C++17}} expected-note 0-1{{noexcept}}
  void (A::*af2)() throw() = &A::f; // expected-error-re {{{{not superset|different exception spec}}}}

#if __cplusplus >= 201103L
  static_assert(noexcept(A().g()), "");
#endif

  // Likewise, they're instantiated separately from an enclosing class template.
  template<typename U>
  struct B {
    void f() throw(T, typename U::type) {} // expected-error 0-1{{C++17}} expected-note 0-1{{noexcept}}
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

  void (B<P>::*bpf1)() throw(B<P>::T, int) = &B<P>::f; // expected-error 0-1{{C++17}} expected-note 0-1{{noexcept}}
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
  void (B<P>::*bpf2)() throw(int) = &B<P>::f; // expected-error 0-1{{C++17}} expected-note 0-1{{noexcept}}
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

  template<typename T> int f() throw(typename T::error) { return 0; } // expected-error 1-4{{prior to '::'}} expected-note 0-1{{prior to '::'}} expected-note 0-1{{requested here}}
#if __cplusplus > 201402L
    // expected-error@-2 0-1{{C++17}} expected-note@-2 0-1{{noexcept}}
#endif
  // An exception-specification is needed even if the function is only used in
  // an unevaluated operand.
  int f1 = sizeof(f<int>()); // expected-note {{instantiation of}}
#if __cplusplus >= 201103L
  decltype(f<char>()) f2; // expected-note {{instantiation of}}
  bool f3 = noexcept(f<float>()); // expected-note {{instantiation of}}
#endif
  // In C++17 onwards, substituting explicit template arguments into the
  // function type substitutes into the exception specification (because it's
  // part of the type). In earlier languages, we don't notice there's a problem
  // until we've already started to instantiate.
  template int f<short>();
#if __cplusplus >= 201703L
  // expected-error@-2 {{does not refer to a function template}}
#else
  // expected-note@-4 {{instantiation of}}
#endif

  template<typename T> struct C {
    C() throw(typename T::type); // expected-error 1-2{{prior to '::'}}
#if __cplusplus > 201402L
    // expected-error@-2 0-1{{C++17}} expected-note@-2 0-1{{noexcept}}
#endif
  };
  struct D : C<void> {}; // ok
#if __cplusplus < 201103L
  // expected-note@-2 {{instantiation of}}
#endif
  void f(D &d) { d = d; } // ok

  struct E : C<int> {}; // expected-note {{in instantiation of}}
#if __cplusplus >= 201103L
  E e; // expected-note {{needed here}}
#endif
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

namespace dr1347 { // dr1347: yes
  auto x = 5, *y = &x; // expected-error 0-1{{extension}}
  auto z = y, *q = y; // expected-error {{'auto' deduced as 'int *' in declaration of 'z' and deduced as 'int' in declaration of 'q'}} expected-error 0-1{{extension}}
#if __cplusplus >= 201103L
  auto a = 5, b = {1, 2}; // expected-error {{'auto' deduced as 'int' in declaration of 'a' and deduced as 'std::initializer_list<int>' in declaration of 'b'}}
  auto (*fp)(int) -> int, i = 0; // expected-error {{declaration with trailing return type must be the only declaration in its group}}
#endif
}

namespace dr1358 { // dr1358: yes
#if __cplusplus >= 201103L
  struct Lit { constexpr operator int() const { return 0; } };
  struct NonLit { NonLit(); operator int(); }; // expected-note 2{{no constexpr constructors}}
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
    constexpr B(NonLit u) : member(u) {} // expected-error {{not a literal type}}
    constexpr NonLit f(NonLit u) const { return NonLit(); } // expected-error {{not a literal type}}
  };
#endif
}

namespace dr1359 { // dr1359: 3.5
#if __cplusplus >= 201103L
  union A { constexpr A() = default; };
  union B { constexpr B() = default; int a; }; // expected-error {{not constexpr}} expected-note 2{{candidate}}
  union C { constexpr C() = default; int a, b; }; // expected-error {{not constexpr}} expected-note 2{{candidate}}
  struct X { constexpr X() = default; union {}; }; // expected-error {{does not declare anything}}
  struct Y { constexpr Y() = default; union { int a; }; }; // expected-error {{not constexpr}} expected-note 2{{candidate}}

  constexpr A a = A();
  constexpr B b = B(); // expected-error {{no matching}}
  constexpr C c = C(); // expected-error {{no matching}}
  constexpr X x = X();
  constexpr Y y = Y(); // expected-error {{no matching}}
#endif
}

namespace dr1388 { // dr1388: 4
  template<typename A, typename ...T> void f(T..., A); // expected-note 1+{{candidate}} expected-error 0-1{{C++11}}
  template<typename ...T> void g(T..., int); // expected-note 1+{{candidate}} expected-error 0-1{{C++11}}
  template<typename ...T, typename A> void h(T..., A); // expected-note 1+{{candidate}} expected-error 0-1{{C++11}}

  void test_f() { 
    f(0); // ok, trailing parameter pack deduced to empty
    f(0, 0); // expected-error {{no matching}}
    f<int>(0);
    f<int>(0, 0); // expected-error {{no matching}}
    f<int, int>(0, 0);
    f<int, int, int>(0, 0); // expected-error {{no matching}}

    g(0);
    g(0, 0); // expected-error {{no matching}}
    g<>(0);
    g<int>(0); // expected-error {{no matching}}
    g<int>(0, 0);

    h(0);
    h(0, 0); // expected-error {{no matching}}
    h<int>(0, 0);
    h<int, int>(0, 0); // expected-error {{no matching}}
  }

  // A non-trailing parameter pack is still a non-deduced context, even though
  // we know exactly how many arguments correspond to it.
  template<typename T, typename U> struct pair {};
  template<typename ...T> struct tuple { typedef char type; }; // expected-error 0-2{{C++11}}
  template<typename ...T, typename ...U> void f_pair_1(pair<T, U>..., int); // expected-error 0-2{{C++11}} expected-note {{different lengths (2 vs. 0)}}
  template<typename ...T, typename U> void f_pair_2(pair<T, char>..., U); // expected-error 0-2{{C++11}}
  template<typename ...T, typename ...U> void f_pair_3(pair<T, U>..., tuple<U...>); // expected-error 0-2{{C++11}} expected-note {{different lengths (2 vs. 1)}}
  template<typename ...T> void f_pair_4(pair<T, char>..., T...); // expected-error 0-2{{C++11}} expected-note {{<int, long> vs. <int, long, const char *>}}
  void g(pair<int, char> a, pair<long, char> b, tuple<char, char> c) {
    f_pair_1<int, long>(a, b, 0); // expected-error {{no match}}
    f_pair_2<int, long>(a, b, 0);
    f_pair_3<int, long>(a, b, c);
    f_pair_3<int, long>(a, b, tuple<char>()); // expected-error {{no match}}
    f_pair_4<int, long>(a, b, 0, 0L);
    f_pair_4<int, long>(a, b, 0, 0L, "foo"); // expected-error {{no match}}
  }
}

namespace dr1391 { // dr1391: partial
  struct A {}; struct B : A {};
  template<typename T> struct C { C(int); typename T::error error; }; // expected-error 2{{'::'}}
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
    b<int>(0, 0); // expected-note {{instantiation of}} expected-note {{while substituting}}
  }

  template<typename T> struct Id { typedef T type; };
  template<typename T> void c(T, typename Id<C<T> >::type);
  void test_c() {
    // Implicit conversion sequences for dependent types are checked later.
    c(0.0, 0); // expected-note {{instantiation of}}
  }

  namespace partial_ordering {
    // FIXME: Second template should be considered more specialized because non-dependent parameter is ignored.
    template<typename T> int a(T, short) = delete; // expected-error 0-1{{extension}} expected-note {{candidate}}
    template<typename T> int a(T*, char); // expected-note {{candidate}}
    int test_a = a((int*)0, 0); // FIXME: expected-error {{ambiguous}}

    // FIXME: Second template should be considered more specialized:
    // deducing #1 from #2 ignores the second P/A pair, so deduction succeeds,
    // deducing #2 from #1 fails to deduce T, so deduction fails.
    template<typename T> int b(T, int) = delete; // expected-error 0-1{{extension}} expected-note {{candidate}}
    template<typename T, typename U> int b(T*, U); // expected-note {{candidate}}
    int test_b = b((int*)0, 0); // FIXME: expected-error {{ambiguous}}

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

namespace dr1399 { // dr1399: dup 1388
  template<typename ...T> void f(T..., int, T...) {} // expected-note {{candidate}} expected-error 0-1{{C++11}}
  void g() {
    f(0);
    f<int>(0, 0, 0);
    f(0, 0, 0); // expected-error {{no match}}
  }
}
