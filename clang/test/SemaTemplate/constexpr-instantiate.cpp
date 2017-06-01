// RUN: %clang_cc1 -std=c++11 -verify %s

namespace UseBeforeDefinition {
  struct A {
    template<typename T> static constexpr T get() { return T(); }
    // ok, not a constant expression.
    int n = get<int>();
  };

  // ok, constant expression.
  constexpr int j = A::get<int>();

  template<typename T> constexpr int consume(T);
  // ok, not a constant expression.
  const int k = consume(0); // expected-note {{here}}

  template<typename T> constexpr int consume(T) { return 0; }
  // ok, constant expression.
  constexpr int l = consume(0);

  constexpr int m = k; // expected-error {{constant expression}} expected-note {{initializer of 'k'}}
}

namespace IntegralConst {
  template<typename T> constexpr T f(T n) { return n; }
  enum E {
    v = f(0), w = f(1) // ok
  };
  static_assert(w == 1, "");

  char arr[f('x')]; // ok
  static_assert(sizeof(arr) == 'x', "");
}

namespace ConvertedConst {
  template<typename T> constexpr T f(T n) { return n; }
  int f() {
    switch (f()) {
      case f(4): return 0;
    }
    return 1;
  }
}

namespace OverloadResolution {
  template<typename T> constexpr T f(T t) { return t; }

  template<int n> struct S { };

  template<typename T> auto g(T t) -> S<f(sizeof(T))> &;
  char &f(...);

  template<typename T> auto h(T t[f(sizeof(T))]) -> decltype(&*t) {
    return t;
  }

  S<4> &k = g(0);
  int *p, *q = h(p);
}

namespace DataMember {
  template<typename T> struct S { static const int k; };
  const int n = S<int>::k; // expected-note {{here}}
  template<typename T> const int S<T>::k = 0;
  constexpr int m = S<int>::k; // ok
  constexpr int o = n; // expected-error {{constant expression}} expected-note {{initializer of 'n'}}
}

namespace Reference {
  const int k = 5;
  template<typename T> struct S {
    static volatile int &r;
  };
  template<typename T> volatile int &S<T>::r = const_cast<volatile int&>(k);
  constexpr int n = const_cast<int&>(S<int>::r);
  static_assert(n == 5, "");
}

namespace Unevaluated {
  // We follow the current proposed resolution of core issue 1581: a constexpr
  // function template specialization requires a definition if:
  //  * it is odr-used, or would be odr-used except that it appears within the
  //    definition of a template, or
  //  * it is used within a braced-init-list, where it may be necessary for
  //    detecting narrowing conversions.
  //
  // We apply this both for instantiating constexpr function template
  // specializations and for implicitly defining defaulted constexpr special
  // member functions.
  //
  // FIXME: None of this is required by the C++ standard yet. The rules in this
  //        area are subject to change.
  namespace NotConstexpr {
    template<typename T> struct S {
      S() : n(0) {}
      S(const S&) : n(T::error) {}
      int n;
    };
    struct U : S<int> {};
    decltype(U(U())) u;
  }
  namespace Constexpr {
    template<typename T> struct S {
      constexpr S() : n(0) {}
      constexpr S(const S&) : n(T::error) {}
      int n;
    };
    struct U : S<int> {};
    decltype(U(U())) u;
  }
  namespace ConstexprList {
    template<int N> struct S {
      constexpr S() : n(0) {
        static_assert(N >= 0, "");
      }
      constexpr operator int() const { return 0; }
      int n;
    };
    struct U : S<0> {};
    // ok, trigger instantiation within a list
    decltype(char{U()}) t0;
    decltype(new char{S<1>()}) t1; // expected-warning {{side effects}}
    decltype((char){S<2>()}) t2;
    decltype(+(char[1]){{S<3>()}}) t3;
    // do not trigger instantiation outside a list
    decltype(char(S<-1>())) u1;
    decltype(new char(S<-2>())) u2; // expected-warning {{side effects}}
    decltype((char)(S<-3>())) u3;
  }

  namespace PR11851_Comment0 {
    template<int x> constexpr int f() { return x; }
    template<int i> void ovf(int (&x)[f<i>()]);
    void f() { int x[10]; ovf<10>(x); }
  }

  namespace PR11851_Comment1 {
    template<typename T>
    constexpr bool Integral() {
      return true;
    }
    template<typename T, bool Int = Integral<T>()>
    struct safe_make_unsigned {
      typedef T type;
    };
    template<typename T>
    using Make_unsigned = typename safe_make_unsigned<T>::type;
    template <typename T>
    struct get_distance_type {
      using type = int;
    };
    template<typename R>
    auto size(R) -> Make_unsigned<typename get_distance_type<R>::type>;
    auto check() -> decltype(size(0));
  }

  namespace PR11851_Comment6 {
    template<int> struct foo {};
    template<class> constexpr int bar() { return 0; }
    template<class T> foo<bar<T>()> foobar();
    auto foobar_ = foobar<int>();
  }

  namespace PR11851_Comment9 {
    struct S1 {
      constexpr S1() {}
      constexpr operator int() const { return 0; }
    };
    int k1 = sizeof(short{S1(S1())});

    struct S2 {
      constexpr S2() {}
      constexpr operator int() const { return 123456; }
    };
    int k2 = sizeof(short{S2(S2())}); // expected-error {{cannot be narrowed}} expected-note {{insert an explicit cast to silence this issue}}
  }

  namespace PR12288 {
    template <typename> constexpr bool foo() { return true; }
    template <bool> struct bar {};
    template <typename T> bar<foo<T>()> baz() { return bar<foo<T>()>(); }
    int main() { baz<int>(); }
  }

  namespace PR13423 {
    template<bool, typename> struct enable_if {};
    template<typename T> struct enable_if<true, T> { using type = T; };

    template<typename T> struct F {
      template<typename U>
      static constexpr bool f() { return sizeof(T) < U::size; }

      template<typename U>
      static typename enable_if<f<U>(), void>::type g() {} // expected-note {{disabled by 'enable_if'}}
    };

    struct U { static constexpr int size = 2; };

    void h() { F<char>::g<U>(); }
    void i() { F<int>::g<U>(); } // expected-error {{no matching function}}
  }

  namespace PR14203 {
    struct duration { constexpr duration() {} };

    template <typename>
    void sleep_for() {
      constexpr duration max = duration();
    }
  }

  // For variables, we instantiate when they are used in a context in which
  // evaluation could be required (odr-used, used in a template whose
  // instantiations would odr-use, or used in list initialization), if they
  // can be used as a constant (const integral or constexpr).
  namespace Variables {
    template<int N> struct A {
      static const int k;
      static int n;
    };
    template<const int *N> struct B {};
    template<int N> constexpr int A<N>::k = *(int[N]){N}; // expected-error 1+{{negative}}
    template<int N> int A<N>::n = *(int[N]){0};

    template <typename> void f() {
      (void)A<-1>::n; // ok
      (void)A<-1>::k; // expected-note {{instantiation of }}
      B<&A<-2>::n> b1; // ok
      B<&A<-2>::k> b2; // expected-note {{instantiation of }}
    };

    decltype(A<-3>::k) d1 = 0; // ok
    decltype(char{A<-4>::k}) d2 = 0; // expected-note {{instantiation of }} expected-error {{narrow}} expected-note {{cast}}
    decltype(char{A<1>::k}) d3 = 0; // ok
    decltype(char{A<1 + (unsigned char)-1>::k}) d4 = 0; // expected-error {{narrow}} expected-note {{cast}}
  }
}

namespace NoInstantiationWhenSelectingOverload {
  // Check that we don't instantiate conversion functions when we're checking
  // for the existence of an implicit conversion sequence, only when a function
  // is actually chosen by overload resolution.
  struct S {
    template<typename T> constexpr S(T) : n(T::error) {} // expected-error {{no members}}
    int n;
  };

  constexpr int f(S) { return 0; }
  constexpr int f(int) { return 0; }

  void g() { f(0); }
  void h() { (void)sizeof(char{f(0)}); }
  void i() { (void)sizeof(char{f("oops")}); } // expected-note {{instantiation of}}
}

namespace PR20090 {
  template <typename T> constexpr T fact(T n) {
    return n == 0 ? 1 : [=] { return n * fact(n - 1); }();
  }
  static_assert(fact(0) == 1, "");
}
