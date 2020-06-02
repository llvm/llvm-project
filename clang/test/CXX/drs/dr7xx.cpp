// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++14 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++17 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++2a %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

namespace dr705 { // dr705: yes
  namespace N {
    struct S {};
    void f(S); // expected-note {{declared here}}
  }

  void g() {
    N::S s;
    f(s);      // ok
    (f)(s);    // expected-error {{use of undeclared}}
  }
}

namespace dr712 { // dr712: partial
  void use(int);
  void f() {
    const int a = 0; // expected-note 5{{here}}
    struct X {
      void g(bool cond) {
        use(a);
        use((a));
        use(cond ? a : a);
        use((cond, a)); // expected-warning 2{{unused}} FIXME: should only warn once

        (void)a; // FIXME: expected-error {{declared in enclosing}}
        (void)(a); // FIXME: expected-error {{declared in enclosing}}
        (void)(cond ? a : a); // FIXME: expected-error 2{{declared in enclosing}}
        (void)(cond, a); // FIXME: expected-error {{declared in enclosing}} expected-warning {{unused}}
      }
    };
  }

#if __cplusplus >= 201103L
  void g() {
    struct A { int n; };
    constexpr A a = {0}; // expected-note 2{{here}}
    struct X {
      void g(bool cond) {
        use(a.n);
        use(a.*&A::n);

        (void)a.n; // FIXME: expected-error {{declared in enclosing}}
        (void)(a.*&A::n); // FIXME: expected-error {{declared in enclosing}}
      }
    };
  }
#endif
}

namespace dr727 { // dr727: partial
  struct A {
    template<typename T> struct C; // expected-note 6{{here}}
    template<typename T> void f(); // expected-note {{here}}
    template<typename T> static int N; // expected-error 0-1{{C++14}} expected-note 6{{here}}

    template<> struct C<int>;
    template<> void f<int>();
    template<> static int N<int>;

    template<typename T> struct C<T*>;
    template<typename T> static int N<T*>;

    struct B {
      template<> struct C<float>; // expected-error {{not in class 'A' or an enclosing namespace}}
      template<> void f<float>(); // expected-error {{no function template matches}}
      template<> static int N<float>; // expected-error {{not in class 'A' or an enclosing namespace}}

      template<typename T> struct C<T**>; // expected-error {{not in class 'A' or an enclosing namespace}}
      template<typename T> static int N<T**>; // expected-error {{not in class 'A' or an enclosing namespace}}

      template<> struct A::C<double>; // expected-error {{not in class 'A' or an enclosing namespace}}
      template<> void A::f<double>(); // expected-error {{no function template matches}} expected-error {{cannot have a qualified name}}
      template<> static int A::N<double>; // expected-error {{not in class 'A' or an enclosing namespace}} expected-error {{cannot have a qualified name}}

      template<typename T> struct A::C<T***>; // expected-error {{not in class 'A' or an enclosing namespace}}
      template<typename T> static int A::N<T***>; // expected-error {{not in class 'A' or an enclosing namespace}} expected-error {{cannot have a qualified name}}
    };
  };

  template<> struct A::C<char>;
  template<> void A::f<char>();
  template<> int A::N<char>;

  template<typename T> struct A::C<T****>;
  template<typename T> int A::N<T****>;

  namespace C {
    template<> struct A::C<long>; // expected-error {{not in class 'A' or an enclosing namespace}}
    template<> void A::f<long>(); // expected-error {{not in class 'A' or an enclosing namespace}}
    template<> int A::N<long>; // expected-error {{not in class 'A' or an enclosing namespace}}

    template<typename T> struct A::C<T*****>; // expected-error {{not in class 'A' or an enclosing namespace}}
    template<typename T> int A::N<T*****>; // expected-error {{not in class 'A' or an enclosing namespace}}
  }

  template<typename>
  struct D {
    template<typename T> struct C { typename T::error e; }; // expected-error {{no members}}
    template<typename T> void f() { T::error; } // expected-error {{no members}}
    template<typename T> static const int N = T::error; // expected-error {{no members}} expected-error 0-1{{C++14}}

    template<> struct C<int> {};
    template<> void f<int>() {}
    template<> static const int N<int>;

    template<typename T> struct C<T*> {};
    template<typename T> static const int N<T*>;
  };

  void d(D<int> di) {
    D<int>::C<int>();
    di.f<int>();
    int a = D<int>::N<int>;

    D<int>::C<int*>();
    int b = D<int>::N<int*>;

    D<int>::C<float>(); // expected-note {{instantiation of}}
    di.f<float>(); // expected-note {{instantiation of}}
    int c = D<int>::N<float>; // expected-note {{instantiation of}}
  }

  namespace mixed_inner_outer_specialization {
#if __cplusplus >= 201103L
    template<int> struct A {
      template<int> constexpr int f() const { return 1; }
      template<> constexpr int f<0>() const { return 2; }
    };
    template<> template<int> constexpr int A<0>::f() const { return 3; }
    template<> template<> constexpr int A<0>::f<0>() const { return 4; }
    static_assert(A<1>().f<1>() == 1, "");
    static_assert(A<1>().f<0>() == 2, "");
    static_assert(A<0>().f<1>() == 3, "");
    static_assert(A<0>().f<0>() == 4, "");
#endif

#if __cplusplus >= 201402L
    template<int> struct B {
      template<int> static const int u = 1;
      template<> static const int u<0> = 2; // expected-note {{here}}

      // Note that in C++17 onwards, these are implicitly inline, and so the
      // initializer of v<0> is not instantiated with the declaration. In
      // C++14, v<0> is a non-defining declaration and its initializer is
      // instantiated with the class.
      template<int> static constexpr int v = 1;
      template<> static constexpr int v<0> = 2; // #v0

      template<int> static const inline int w = 1; // expected-error 0-1{{C++17 extension}}
      template<> static const inline int w<0> = 2; // expected-error 0-1{{C++17 extension}}
    };

    template<> template<int> constexpr int B<0>::u = 3;
    template<> template<> constexpr int B<0>::u<0> = 4; // expected-error {{already has an initializer}}

    template<> template<int> constexpr int B<0>::v = 3;
    template<> template<> constexpr int B<0>::v<0> = 4;
#if __cplusplus < 201702L
    // expected-error@-2 {{already has an initializer}}
    // expected-note@#v0 {{here}}
#endif

    template<> template<int> constexpr int B<0>::w = 3;
    template<> template<> constexpr int B<0>::w<0> = 4;

    static_assert(B<1>().u<1> == 1, "");
    static_assert(B<1>().u<0> == 2, "");
    static_assert(B<0>().u<1> == 3, "");

    static_assert(B<1>().v<1> == 1, "");
    static_assert(B<1>().v<0> == 2, "");
    static_assert(B<0>().v<1> == 3, "");
    static_assert(B<0>().v<0> == 4, "");
#if __cplusplus < 201702L
    // expected-error@-2 {{failed}}
#endif

    static_assert(B<1>().w<1> == 1, "");
    static_assert(B<1>().w<0> == 2, "");
    static_assert(B<0>().w<1> == 3, "");
    static_assert(B<0>().w<0> == 4, "");
#endif
  }

  template<typename T, typename U> struct Collision {
    // FIXME: Missing diagnostic for duplicate function explicit specialization declaration.
    template<typename> int f1();
    template<> int f1<T>();
    template<> int f1<U>();

    // FIXME: Missing diagnostic for fucntion redefinition!
    template<typename> int f2();
    template<> int f2<T>() {}
    template<> int f2<U>() {}

    template<typename> static int v1; // expected-error 0-1{{C++14 extension}}
    template<> static int v1<T>; // expected-note {{previous}}
    template<> static int v1<U>; // expected-error {{duplicate member}}

    template<typename> static inline int v2; // expected-error 0-1{{C++17 extension}} expected-error 0-1{{C++14 extension}}
    template<> static inline int v2<T>;      // expected-error 0-1{{C++17 extension}} expected-note {{previous}}
    template<> static inline int v2<U>;      // expected-error 0-1{{C++17 extension}} expected-error {{duplicate member}}

    // FIXME: Missing diagnostic for duplicate class explicit specialization.
    template<typename> struct S1;
    template<> struct S1<T>;
    template<> struct S1<U>;

    template<typename> struct S2;
    template<> struct S2<T> {}; // expected-note {{previous}}
    template<> struct S2<U> {}; // expected-error {{redefinition}}
  };
  Collision<int, int> c; // expected-note {{in instantiation of}}
}

// dr777 superseded by dr2233
