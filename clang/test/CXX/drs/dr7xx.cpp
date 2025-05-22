// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++98 %s -verify=expected,cxx98-14,cxx98-11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++11 %s -verify=expected,cxx98-14,cxx98-11,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++14 %s -verify=expected,cxx98-14,since-cxx14,since-cxx11,cxx14 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++17 %s -verify=expected,since-cxx14,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++2a %s -verify=expected,since-cxx14,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors

namespace dr705 { // dr705: yes
  namespace N {
    struct S {};
    void f(S); // #dr705-f
  }

  void g() {
    N::S s;
    f(s);      // ok
    (f)(s);
    // expected-error@-1 {{use of undeclared identifier 'f'}}
    //   expected-note@#dr705-f {{'N::f' declared here}}
  }
}

namespace dr712 { // dr712: partial
  void use(int);
  void f() {
    const int a = 0; // #dr712-f-a
    struct X {
      void g(bool cond) {
        use(a);
        use((a));
        use(cond ? a : a);
        // FIXME: should only warn once
        use((cond, a));
        // expected-warning@-1 {{left operand of comma operator has no effect}}
        // expected-warning@-2 {{left operand of comma operator has no effect}}

        (void)a;
        // expected-error@-1 {{reference to local variable 'a' declared in enclosing function 'dr712::f'}} FIXME
        //   expected-note@#dr712-f-a {{'a' declared here}}
        (void)(a);
        // expected-error@-1 {{reference to local variable 'a' declared in enclosing function 'dr712::f'}} FIXME
        //   expected-note@#dr712-f-a {{'a' declared here}}
        (void)(cond ? a : a); // #dr712-ternary
        // expected-error@#dr712-ternary {{reference to local variable 'a' declared in enclosing function 'dr712::f'}} FIXME
        //   expected-note@#dr712-f-a {{'a' declared here}}
        // expected-error@#dr712-ternary {{reference to local variable 'a' declared in enclosing function 'dr712::f'}} FIXME
        //   expected-note@#dr712-f-a {{'a' declared here}}
        (void)(cond, a); // #dr712-comma
        // expected-error@-1 {{reference to local variable 'a' declared in enclosing function 'dr712::f'}} FIXME
        //   expected-note@#dr712-f-a {{'a' declared here}}
        // expected-warning@#dr712-comma {{left operand of comma operator has no effect}}
      }
    };
  }

#if __cplusplus >= 201103L
  void g() {
    struct A { int n; };
    constexpr A a = {0};  // #dr712-g-a
    struct X {
      void g(bool cond) {
        use(a.n);
        use(a.*&A::n);

        (void)a.n;
        // since-cxx11-error@-1 {{reference to local variable 'a' declared in enclosing function 'dr712::g'}} FIXME
        //   since-cxx11-note@#dr712-g-a {{'a' declared here}}
        (void)(a.*&A::n);
        // since-cxx11-error@-1 {{reference to local variable 'a' declared in enclosing function 'dr712::g'}} FIXME
        //   since-cxx11-note@#dr712-g-a {{'a' declared here}}
      }
    };
  }
#endif
}

namespace dr727 { // dr727: partial
  struct A {
    template<typename T> struct C; // #dr727-C
    template<typename T> void f(); // #dr727-f
    template<typename T> static int N; // #dr727-N
    // cxx98-11-error@-1 {{variable templates are a C++14 extension}}

    template<> struct C<int>;
    template<> void f<int>();
    template<> static int N<int>;

    template<typename T> struct C<T*>;
    template<typename T> static int N<T*>;

    struct B {
      template<> struct C<float>;
      // expected-error@-1 {{class template specialization of 'C' not in class 'A' or an enclosing namespace}}
      //   expected-note@#dr727-C {{explicitly specialized declaration is here}}
      template<> void f<float>();
      // expected-error@-1 {{no function template matches function template specialization 'f'}}
      template<> static int N<float>;
      // expected-error@-1 {{variable template specialization of 'N' not in class 'A' or an enclosing namespace}}
      //   expected-note@#dr727-N {{explicitly specialized declaration is here}}

      template<typename T> struct C<T**>;
      // expected-error@-1 {{class template partial specialization of 'C' not in class 'A' or an enclosing namespace}}
      //   expected-note@#dr727-C {{explicitly specialized declaration is here}}
      template<typename T> static int N<T**>;
      // expected-error@-1 {{variable template partial specialization of 'N' not in class 'A' or an enclosing namespace}}
      //   expected-note@#dr727-N {{explicitly specialized declaration is here}}

      template<> struct A::C<double>;
      // expected-error@-1 {{non-friend class member 'C' cannot have a qualified name}}
      // expected-error@-2 {{class template specialization of 'C' not in class 'A' or an enclosing namespace}}
      //   expected-note@#dr727-C {{explicitly specialized declaration is here}}
      template<> void A::f<double>();
      // expected-error@-1 {{o function template matches function template specialization 'f'}}
      // expected-error@-2 {{non-friend class member 'f' cannot have a qualified name}}
      template<> static int A::N<double>;
      // expected-error@-1 {{non-friend class member 'N' cannot have a qualified name}}
      // expected-error@-2 {{variable template specialization of 'N' not in class 'A' or an enclosing namespace}}
      //   expected-note@#dr727-N {{explicitly specialized declaration is here}}

      template<typename T> struct A::C<T***>;
      // expected-error@-1 {{non-friend class member 'C' cannot have a qualified name}}
      // expected-error@-2 {{class template partial specialization of 'C' not in class 'A' or an enclosing namespace}}
      //   expected-note@#dr727-C {{explicitly specialized declaration is here}}
      template<typename T> static int A::N<T***>;
      // expected-error@-1 {{non-friend class member 'N' cannot have a qualified name}}
      // expected-error@-2 {{variable template partial specialization of 'N' not in class 'A' or an enclosing namespace}}
      //   expected-note@#dr727-N {{explicitly specialized declaration is here}}
    };
  };

  template<> struct A::C<char>;
  template<> void A::f<char>();
  template<> int A::N<char>;

  template<typename T> struct A::C<T****>;
  template<typename T> int A::N<T****>;

  namespace C {
    template<> struct A::C<long>;
    // expected-error@-1 {{class template specialization of 'C' not in class 'A' or an enclosing namespace}}
    //   expected-note@#dr727-C {{explicitly specialized declaration is here}}
    template<> void A::f<long>();
    // expected-error@-1 {{function template specialization of 'f' not in class 'A' or an enclosing namespace}}
    //   expected-note@#dr727-f {{explicitly specialized declaration is here}}
    template<> int A::N<long>;
    // expected-error@-1 {{variable template specialization of 'N' not in class 'A' or an enclosing namespace}}
    //   expected-note@#dr727-N {{explicitly specialized declaration is here}}

    template<typename T> struct A::C<T*****>;
    // expected-error@-1 {{class template partial specialization of 'C' not in class 'A' or an enclosing namespace}}
    //   expected-note@#dr727-C {{explicitly specialized declaration is here}}
    template<typename T> int A::N<T*****>;
    // expected-error@-1 {{variable template partial specialization of 'N' not in class 'A' or an enclosing namespace}}
    //   expected-note@#dr727-N {{explicitly specialized declaration is here}}
  }

  template<typename>
  struct D {
    template<typename T> struct C { typename T::error e; };
    // expected-error@-1 {{type 'float' cannot be used prior to '::' because it has no members}}
    //   expected-note@#dr727-C-float {{in instantiation of template class 'dr727::D<int>::C<float>' requested here}}
    template<typename T> void f() { T::error; }
    // expected-error@-1 {{type 'float' cannot be used prior to '::' because it has no members}}
    //   expected-note@#dr727-f-float {{in instantiation of function template specialization 'dr727::D<int>::f<float>' requested here}}
    template<typename T> static const int N = T::error;
    // cxx98-11-error@-1 {{variable templates are a C++14 extension}}
    // expected-error@-2 {{type 'float' cannot be used prior to '::' because it has no members}}
    //   expected-note@#dr727-N-float {{in instantiation of static data member 'dr727::D<int>::N<float>' requested here}}

    template<> struct C<int> {};
    template<> void f<int>() {}
    template<> static const int N<int>;

    template<typename T> struct C<T*> {};
    template<typename T> static const int N<T*>;

    template<typename>
    struct E {
      template<> void f<void>() {}
      // expected-error@-1 {{no candidate function template was found for dependent member function template specialization}}
    };
  };

  void d(D<int> di) {
    D<int>::C<int>();
    di.f<int>();
    int a = D<int>::N<int>;

    D<int>::C<int*>();
    int b = D<int>::N<int*>;

    D<int>::C<float>(); // #dr727-C-float
    di.f<float>(); // #dr727-f-float
    int c = D<int>::N<float>; // #dr727-N-float
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
      template<> static const int u<0> = 2; // #dr727-u0

      // Note that in C++17 onwards, these are implicitly inline, and so the
      // initializer of v<0> is not instantiated with the declaration. In
      // C++14, v<0> is a non-defining declaration and its initializer is
      // instantiated with the class.
      template<int> static constexpr int v = 1;
      template<> static constexpr int v<0> = 2; // #dr727-v0

      template<int> static const inline int w = 1;
      // cxx14-error@-1 {{inline variables are a C++17 extension}}
      template<> static const inline int w<0> = 2;
      // cxx14-error@-1 {{inline variables are a C++17 extension}}
    };

    template<> template<int> constexpr int B<0>::u = 3;
    template<> template<> constexpr int B<0>::u<0> = 4;
    // since-cxx14-error@-1 {{static data member 'u' already has an initializer}}
    //   since-cxx14-note@#dr727-u0 {{previous initialization is here}}

    template<> template<int> constexpr int B<0>::v = 3;
    template<> template<> constexpr int B<0>::v<0> = 4;
    // cxx14-error@-1 {{static data member 'v' already has an initializer}}
    //   cxx14-note@#dr727-v0 {{previous initialization is here}}

    template<> template<int> constexpr int B<0>::w = 3;
    template<> template<> constexpr int B<0>::w<0> = 4;

    static_assert(B<1>().u<1> == 1, "");
    static_assert(B<1>().u<0> == 2, "");
    static_assert(B<0>().u<1> == 3, "");

    static_assert(B<1>().v<1> == 1, "");
    static_assert(B<1>().v<0> == 2, "");
    static_assert(B<0>().v<1> == 3, "");
    static_assert(B<0>().v<0> == 4, "");
    // cxx14-error@-1 {{static assertion failed due to requirement 'dr727::mixed_inner_outer_specialization::B<0>().v<0> == 4'}}
    //   cxx14-note@-2 {{expression evaluates to '2 == 4'}}

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

    template<typename> static int v1;
    // cxx98-11-error@-1 {{variable templates are a C++14 extension}}
    template<> static int v1<T>; // #dr727-v1-T
    template<> static int v1<U>;
    // expected-error@-1 {{duplicate member 'v1'}}
    //   expected-note@#dr727-Collision-int-int {{in instantiation of template class 'dr727::Collision<int, int>' requested here}}
    //   expected-note@#dr727-v1-T {{previous}}

    template<typename> static inline int v2;
    // cxx98-11-error@-1 {{variable templates are a C++14 extension}}
    // cxx98-14-error@-2 {{inline variables are a C++17 extension}}
    template<> static inline int v2<T>; // #dr727-v2-T
    // cxx98-14-error@-1 {{inline variables are a C++17 extension}} 
    template<> static inline int v2<U>;
    // cxx98-14-error@-1 {{inline variables are a C++17 extension}}
    // expected-error@-2 {{duplicate member 'v2'}}
    //   expected-note@#dr727-v2-T {{previous declaration is here}}

    // FIXME: Missing diagnostic for duplicate class explicit specialization.
    template<typename> struct S1;
    template<> struct S1<T>;
    template<> struct S1<U>;

    template<typename> struct S2;
    template<> struct S2<T> {}; // #dr727-S2-T
    template<> struct S2<U> {};
    // expected-error@-1 {{redefinition of 'S2<int>'}}
    //   expected-note@#dr727-S2-T {{previous}}
  };
  Collision<int, int> c; // #dr727-Collision-int-int
}

namespace dr777 { // dr777: 3.7
#if __cplusplus >= 201103L
template <typename... T>
void f(int i = 0, T ...args) {}
void ff() { f(); }

template <typename... T>
void g(int i = 0, T ...args, T ...args2) {}

template <typename... T>
void h(int i = 0, T ...args, int j = 1) {}
#endif
}
