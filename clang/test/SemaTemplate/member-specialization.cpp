// RUN: %clang_cc1 -std=c++17 -verify %s

template<typename T, typename U> struct X {
  template<typename V> const V &as() { return V::error; }
  template<> const U &as<U>() { return u; }
  U u;
};
int f(X<int, int> x) {
  return x.as<int>();
}

namespace ClassScope1 {
  struct A {
    template <class> struct B {
      template <int V> struct C {
        static constexpr auto value = V / 2;
      };
    };
    template <> template <int V> struct B<void>::C {
      static constexpr auto value = V;
    };
    template <> template <int V> struct B<char>::C {
      static constexpr auto value = 2 * V;
    };
  };

  static_assert(A::B<void>::C<2>::value == 2);
  static_assert(A::B<void>::C<3>::value == 3);
  static_assert(A::B<char>::C<2>::value == 4);
  static_assert(A::B<char>::C<3>::value == 6);
  static_assert(A::B<int>::C<10>::value == 5);
  static_assert(A::B<int>::C<20>::value == 10);
} // ClassScope1

namespace DifferentTemplateHeads1 {
  struct A {
    template <class> struct B {
      template <class> struct C {}; // expected-note {{previous template declaration is here}}
    };
    template <> template <int> struct B<void>::C {};
    // expected-error@-1 {{template parameter has a different kind in template redeclaration}}
  };
} // namespace DifferentTemplateHeads1

namespace GH206866 {
  class A {
    template <class> class B {};
    template <> template <class> class B<void>::C {};
    // expected-error@-1 {{out-of-line definition of 'C' does not match any declaration}}
  };
} // namespace GH206866

namespace GH205971 {
  template<class> struct A {};

  template<>
  template<class>
  struct A<int>::B;
  // expected-error@-1 {{out-of-line definition of 'B' does not match any declaration}}

  template<>
  template<class>
  struct A<int>::B;
  // expected-error@-1 {{out-of-line definition of 'B' does not match any declaration}}
} // namespace GH205971
