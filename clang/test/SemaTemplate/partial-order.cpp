// RUN: %clang_cc1 -std=c++26 %s -verify

namespace hana_enable_if_idiom {
  template<bool> struct A {};
  template<typename, typename = A<true>> struct B;
  template<typename T, bool N> struct B<T, A<N>> {};
  template<typename T> struct B<T, A<T::value>> {};
  struct C {
    static const bool value = true;
  };
  B<C> b;
}

namespace GH132562 {
  struct I {
    int v = 0;
  };

  namespace t1 {
    template <I... X> struct A;
    template <I... X>
      requires ((X.v == 0) ||...)
    struct A<X...>;
  } // namespace t1
  namespace t2 {
    template <I... X> struct A; // expected-note {{template is declared here}}
    template <int... X> struct A<X...>;
    // expected-error@-1 {{is not more specialized than the primary template}}
    // expected-note@-2 {{no viable conversion from 'int' to 'I'}}

    template <int... X> struct B; // expected-note {{template is declared here}}
    template <I... X> struct B<X...>;
    // expected-error@-1 {{is not more specialized than the primary template}}
    // expected-note@-2 {{value of type 'const I' is not implicitly convertible to 'int'}}
  } // namespace t2
  namespace t3 {
    struct J {
      int v = 0;
      constexpr J(int v) : v(v) {}
    };
    template <J... X> struct A;
    template <int... X> struct A<X...>;

    template <int... X> struct B; // expected-note {{template is declared here}}
    template <J... X> struct B<X...>;
    // expected-error@-1 {{is not more specialized than the primary template}}
    // expected-note@-2 {{value of type 'const J' is not implicitly convertible to 'int'}}
  } // namespace t3
} // namespace GH132562
