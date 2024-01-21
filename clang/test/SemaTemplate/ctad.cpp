// RUN: %clang_cc1 -std=c++17 -verify %s

namespace pr41427 {
  template <typename T> class A {
  public:
    A(void (*)(T)) {}
  };

  void D(int) {}

  void f() {
    A a(&D);
    using T = decltype(a);
    using T = A<int>;
  }
}

namespace Access {
  struct B {
  protected:
    struct type {};
  };
  template<typename T> struct D : B { // expected-note {{not viable}}
    D(T, typename T::type); // expected-note {{private member}}
  };
  D b = {B(), {}};

  class X {
    using type = int;
  };
  D x = {X(), {}}; // expected-error {{no viable constructor or deduction guide}}

  // Once we implement proper support for dependent nested name specifiers in
  // friends, this should still work.
  class Y {
    template <typename T> friend D<T>::D(T, typename T::type); // expected-warning {{dependent nested name specifier}}
    struct type {};
  };
  D y = {Y(), {}};

  class Z {
    template <typename T> friend class D;
    struct type {};
  };
  D z = {Z(), {}};
}

namespace GH69987 {
template<class> struct X {};
template<class = void> struct X;
X x;

template<class T, class B> struct Y { Y(T); };
template<class T, class B=void> struct Y ;
Y y(1);
};
