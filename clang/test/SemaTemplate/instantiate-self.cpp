// RUN: %clang_cc1 -std=c++11 -verify %s

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

// FIXME: PR12298: Recursive constexpr function template instantiation leads to
// stack overflow.
#if 0
namespace test5 {
  template<typename T> struct A {
    constexpr T f(T k) { return g(k); }
    constexpr T g(T k) {
      return k ? f(k-1)+1 : 0;
    }
  };
  // This should be accepted.
  constexpr int x = A<int>().f(5);
}

namespace test6 {
  template<typename T> constexpr T f(T);
  template<typename T> constexpr T g(T t) {
    typedef int arr[f(T())];
    return t;
  }
  template<typename T> constexpr T f(T t) {
    typedef int arr[g(T())];
    return t;
  }
  // This should be ill-formed.
  int n = f(0);
}

namespace test7 {
  template<typename T> constexpr T g(T t) {
    return t;
  }
  template<typename T> constexpr T f(T t) {
    typedef int arr[g(T())];
    return t;
  }
  // This should be accepted.
  int n = f(0);
}
#endif
