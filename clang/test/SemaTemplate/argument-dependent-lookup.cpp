// RUN: %clang_cc1 -std=c++14 -verify %s
// RUN: %clang_cc1 -std=c++14 -verify %s -DHAVE_UNQUALIFIED_LOOKUP_RESULTS
// expected-no-diagnostics

namespace address_of {
#ifdef HAVE_UNQUALIFIED_LOOKUP_RESULTS
  struct Q {};
  void operator&(Q);
#endif

  template<typename T> struct A {
    static constexpr auto x = &T::value;
  };

  template<typename T> struct B {
    constexpr int operator&() { return 123; }
  };

  template<typename T> struct C {
    static_assert(sizeof(T) == 123, "");
  };

  struct X1 {
    static B<X1> value;
  };
  struct X2 : B<X2> {
    enum E { value };
    friend constexpr int operator&(E) { return 123; }
  };

  struct Y1 {
    C<int> *value;
  };
  struct Y2 {
    C<int> value();
  };

  // ok, uses ADL to find operator&:
  static_assert(A<X1>::x == 123, "");
  static_assert(A<X2>::x == 123, "");

  // ok, does not use ADL so does not instantiate C<T>:
  static_assert(A<Y1>::x == &Y1::value, "");
  static_assert(A<Y2>::x == &Y2::value, "");
}
