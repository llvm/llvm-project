// RUN: %clang_cc1 -std=c++1z -verify %s

// Test that we cope with failure to expand a pack.
template<typename ...T> struct Unexpanded : T... {
  using T::f; // expected-error {{unexpanded}}
  using typename T::type; // expected-error {{unexpanded}}
  template<typename ...U> void g(U ...u) { f(u...); } // expected-error {{undeclared identifier 'f'}}
  void h() {
    Unexpanded<type...> *p; // expected-error {{undeclared identifier 'type'}}
  }
};
void test_Unexpanded() {
  struct A { void f(); }; // expected-note {{must qualify}}
  struct B { void f(int); }; // expected-note {{must qualify}}
  Unexpanded<A, B>().g(0); // expected-note {{instantiation of}}
}

// Test using non-type members from pack of base classes.
template<typename ...T> struct A : T... { // expected-note 2{{candidate}}
  using T::T ...; // expected-note 2{{inherited here}}
  using T::operator() ...;
  using T::operator T* ...;
  using T::h ...;

  void f(int n) { h(n); } // expected-error {{ambiguous}}
  void f(int n, int m) { h(n, m); } // expected-error {{member using declaration 'h' instantiates to an empty pack}}
  void g(int n) { (*this)(n); } // expected-error {{ambiguous}}
  void g(int n, int m) { (*this)(n, m); } // expected-error {{does not provide a call operator}}
};

namespace test_A {
  struct X {
    X();
    X(int); // expected-note {{candidate}}
    void operator()(int); // expected-note 2{{candidate}}
    operator X *();
    void h(int); // expected-note {{candidate}}
  };
  struct Y {
    Y();
    Y(int, int);
    void operator()(int, int);
    operator Y *();
    void h(int, int); // expected-note {{not viable}}
  };
  struct Z {
    Z();
    Z(int); // expected-note {{candidate}}
    void operator()(int); // expected-note 2{{candidate}}
    operator Z *();
    void h(int); // expected-note {{candidate}}
  };

  void f() {
    A<> a;
    a.f(0, 0); // expected-note {{instantiation of}}
    a.g(0, 0); // expected-note {{instantiation of}}

    A<X, Y> axy(0);
    A<X, Y>(0, 0);
    axy.f(0);
    axy.f(0, 0);
    axy.g(0);
    axy.g(0, 0);
    axy(0);
    axy(0, 0);

    A<X, Y, Z>(0); // expected-error {{ambiguous}}
    A<X, Y, Z> axyz(0, 0);
    axyz.f(0); // expected-note {{instantiation of}}
    axyz.f(0, 0);
    axyz.g(0); // expected-note {{instantiation of}}
    axyz.g(0, 0);
    axyz(0); // expected-error {{ambiguous}}
    axyz(0, 0);

    X *x;
    x = a; // expected-error {{incompatible}}
    x = axy;
    x = axyz;
    x = a.operator X*(); // expected-error {{no member}}
    x = axy.operator X*();
    x = axyz.operator X*();

    Z *z;
    z = axyz;
    z = axyz.operator Z*();
  }
}

// Test using pack of non-type members from single base class.
template<typename X, typename Y, typename ...T> struct B : X, Y {
  using X::operator T* ...;
};

namespace test_B {
  struct X { operator int*(); operator float*(); operator char*(); }; // expected-note {{candidate}}
  struct Y { operator int*(); operator float*(); operator char*(); }; // expected-note {{candidate}}
  B<X, Y, int, float> bif;
  int *pi = bif;
  float *pf = bif;
  char *pc = bif; // expected-error {{ambiguous}}
}

// Test using type member from pack of base classes.
template<typename ...T> struct C : T... {
  using typename T::type ...; // expected-error {{target of using declaration conflicts}}
  void f() { type value; } // expected-error {{member using declaration 'type' instantiates to an empty pack}}
};

namespace test_C {
  struct X { typedef int type; };
  struct Y { typedef int type; }; // expected-note {{conflicting}}
  struct Z { typedef float type; }; // expected-note {{target}}

  void f() {
    C<> c;
    c.f(); // expected-note {{instantiation of}}

    C<X, Y> cxy;
    cxy.f();

    C<X, Y, Z> cxyz; // expected-note {{instantiation of}}
    cxyz.f();
  }
}

// Test using pack of non-types at block scope.
template<typename ...T> int fn1() {
  using T::e ...; // expected-error 2{{class member}} expected-note 2{{instead}}
  // expected-error@-1 2{{produces multiple values}}
  return e; // expected-error {{using declaration 'e' instantiates to an empty pack}}
}

namespace test_fn1 {
  struct X { static int e; };
  struct Y { typedef int e; };
  inline namespace P { enum E { e }; }
  inline namespace Q { enum F { e }; }
  void f() {
    fn1<>(); // expected-note {{instantiation of}}
    fn1<X>(); // expected-note {{instantiation of}}
    fn1<Y>(); // expected-note {{instantiation of}}
    fn1<E>();
    fn1<E, F>(); // expected-note {{instantiation of}}
    fn1<E, X>(); // expected-note {{instantiation of}}
  }
}

// Test using pack of types at block scope.
template<typename ...T> void fn2() {
  // This cannot ever be valid: in order for T::type to be a type, T must be a
  // class, and a class member cannot be named by a block-scope using declaration.
  using typename T::type ...; // expected-error {{class member}}
  type x; // expected-error {{unknown type name 'type'}}
}

// Test partial substitution into class-scope pack.
template<typename ...T> auto lambda1() {
  return [](auto x) {
    struct A : T::template X<decltype(x)>... { // expected-note 1+{{instantiation of}}
      using T::template X<decltype(x)>::f ...;
      using typename T::template X<decltype(x)>::type ...;
      void g(int n) { f(n); } // expected-error {{empty pack}} expected-error {{expected 2, have 1}} expected-error {{ambiguous}}
      void h() { type value; } // expected-error {{empty pack}}
    };
    return A();
  };
}

namespace test_lambda1 {
  struct A {
    template<typename> struct X {
      void f(int); // expected-note {{candidate}}
      using type = int;
    };
  };
  struct B {
    template<typename> struct X {
      void f(int, int); // expected-note {{declared here}} expected-note {{not viable}}
      using type = int;
    };
  };
  struct C {
    template<typename> struct X {
      void f(int); // expected-note {{candidate}}
      void f(int, int); // expected-note {{not viable}}
      using type = int;
    };
  };

  void f() {
    lambda1<>() // expected-note 2{{instantiation of}}
      (0)
      // FIXME: This is poor error recovery
      .g(0); // expected-error {{no member named 'g'}}
    lambda1<A>()
      (0)
      .g(0);
    lambda1<B>()
      (0) // expected-note {{instantiation of}}
      .g(0);
    lambda1<A, B, C>()
      (0) // expected-note {{instantiation of}}
      .g(0);
  }
}

namespace p0195r2_example {
  template<typename ...Ts>
  struct Overloader : Ts... {
    using Ts::operator() ...;
  };

  template<typename ...Ts>
  constexpr auto make_overloader(Ts &&...ts) {
    return Overloader<Ts...>{static_cast<Ts&&>(ts)...};
  }

  void test() {
    auto o = make_overloader(
      [&](int &r) -> int & { return r; }, // expected-note {{candidate function}}
      [&](float &r) -> float & { return r; } // expected-note {{candidate function}}
    );
    int a; float f; double d;
    int &ra = o(a);
    float &rf = o(f);
    double &rd = o(d); // expected-error {{no matching function}}
  }
}
