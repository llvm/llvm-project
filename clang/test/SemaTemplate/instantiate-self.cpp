// RUN: %clang_cc1 -std=c++1z -verify -pedantic-errors %s

// Check that we deal with cases where the instantiation of a class template
// recursively requires the instantiation of the same template.
namespace test1 {
  template<typename T> struct A {
    struct B { // expected-note {{not complete until the closing '}'}}
      B b; // expected-error {{has incomplete type 'test1::A<int>::B'}}
    };
    B b; // expected-note {{in instantiation of}}
  };
  A<int> a; // expected-note {{in instantiation of}}
}

namespace test2 {
  template<typename T> struct A {
    struct B {
      struct C {};
      char c[1 + C()]; // expected-error {{invalid operands to binary expression}}
      friend constexpr int operator+(int, C) { return 4; }
    };
    B b; // expected-note {{in instantiation of}}
  };
  A<int> a; // expected-note {{in instantiation of}}
}

namespace test3 {
  // PR12317
  template<typename T> struct A {
    struct B {
      enum { Val = 1 };
      char c[1 + Val]; // ok
    };
    B b;
  };
  A<int> a;
}

namespace test4 {
  template<typename T> struct M { typedef int type; };
  template<typename T> struct A {
    struct B { // expected-note {{not complete until the closing '}'}}
      int k[typename A<typename M<T>::type>::B().k[0] + 1]; // expected-error {{incomplete type}}
    };
    B b; // expected-note {{in instantiation of}}
  };
  A<int> a; // expected-note {{in instantiation of}}
}

// PR12298: Recursive constexpr function template instantiation leads to
// stack overflow.
namespace test5 {
  template<typename T> struct A {
    constexpr T f(T k) { return g(k); }
    constexpr T g(T k) {
      return k ? f(k-1)+1 : 0;
    }
  };
  constexpr int x = A<int>().f(5); // ok
}

namespace test6 {
  template<typename T> constexpr T f(T);
  template<typename T> constexpr T g(T t) {
    typedef int arr[f(T())]; // expected-error {{variable length array}}
    return t;
  }
  template<typename T> constexpr T f(T t) {
    typedef int arr[g(T())]; // expected-error {{zero size array}} expected-note {{instantiation of}}
    return t;
  }
  int n = f(0); // expected-note 2{{instantiation of}}
}

namespace test7 {
  template<typename T> constexpr T g(T t) {
    return t;
  }
  template<typename T> constexpr T f(T t) {
    typedef int arr[g(T() + 1)];
    return t;
  }
  int n = f(0);
}

namespace test8 {
  template<typename T> struct A {
    int n = A{}.n; // expected-error {{default member initializer for 'n' uses itself}} expected-note {{instantiation of default member init}}
  };
  A<int> ai = {}; // expected-note {{instantiation of default member init}}
}

namespace test9 {
  template<typename T> struct A { enum class B; };
  // FIXME: It'd be nice to give the "it has not yet been instantiated" diagnostic here.
  template<typename T> enum class A<T>::B { k = A<T>::B::k2, k2 = k }; // expected-error {{no member named 'k2'}}
  auto k = A<int>::B::k; // expected-note {{in instantiation of}}
}

namespace test10 {
  template<typename T> struct A {
    void f() noexcept(noexcept(f())); // expected-error {{exception specification of 'f' uses itself}} expected-note {{instantiation of}}
  };
  bool b = noexcept(A<int>().f()); // expected-note {{instantiation of}}
}

namespace test11 {
  template<typename T> const int var = var<T>;
  int k = var<int>;

  template<typename T> struct X {
    static const int k = X<T>::k;
  };
  template<typename T> const int X<T>::k;
  int q = X<int>::k;

  template<typename T> struct Y {
    static const int k;
  };
  template<typename T> const int Y<T>::k = Y<T>::k;
  int r = Y<int>::k;
}

namespace test12 {
  template<typename T> int f(T t, int = f(T())) {} // expected-error {{recursive evaluation of default argument}} expected-note {{instantiation of}}
  struct X {};
  int q = f(X()); // expected-note {{instantiation of}}
}

namespace test13 {
  struct A {
    // Cycle via type of non-type template parameter.
    template<typename T, typename T::template W<T>::type U = 0> struct W { using type = int; };
    // Cycle via default template argument.
    template<typename T, typename U = typename T::template X<T>> struct X {};
    template<typename T, int U = T::template Y<T>::value> struct Y { static const int value = 0; };
    template<typename T, template<typename> typename U = T::template Z<T>::template nested> struct Z { template<typename> struct nested; };
  };
  template<typename T> struct Wrap {
    template<typename U> struct W : A::W<T> {};
    template<typename U> struct X : A::X<T> {};
    template<typename U> struct Y : A::Y<T> {};
    template<typename U> struct Z : A::Z<T> {};
  };
  struct B {
    template<typename U> struct W { using type = int; };
    template<typename U> struct X {};
    template<typename U> struct Y { static const int value = 0; };
    template<typename U> struct Z { template<typename> struct nested; };
  };

  A::W<B> awb;
  A::X<B> axb;
  A::Y<B> ayb;
  A::Z<B> azb;

  A::W<Wrap<Wrap<B>>> awwwb;
  A::X<Wrap<Wrap<B>>> axwwb;
  A::Y<Wrap<Wrap<B>>> aywwb;
  A::Z<Wrap<Wrap<B>>> azwwb;

  // FIXME: These tests cause us to use too much stack and crash on a self-hosted debug build.
  // FIXME: Check for recursion here and give a better diagnostic.
#if 0
  A::W<A> awa;
  A::X<A> axa;
  A::Y<A> aya;
  A::Z<A> aza;
#endif
}
