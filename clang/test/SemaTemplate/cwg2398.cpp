// RUN: %clang_cc1 %s -fsyntax-only -std=c++23                                     -verify=expected,new
// RUN: %clang_cc1 %s -fsyntax-only -std=c++23 -fno-relaxed-template-template-args -verify=expected,old

namespace issue1 {
  template<class T, class U = T> class B {};
  template<template<class> class P, class T> void f(P<T>);
  // new-note@-1 {{deduced type 'B<[...], (default) int>' of 1st parameter does not match adjusted type 'B<[...], float>' of argument [with P = B, T = int]}}
  // old-note@-2 2{{template template argument has different template parameters}}

  void g() {
    f(B<int>()); // old-error {{no matching function for call}}
    f(B<int,float>()); // expected-error {{no matching function for call}}
  }
} // namespace issue1

namespace issue2 {
  template<typename> struct match;

  template<template<typename> class t,typename T> struct match<t<T>>;

  template<template<typename,typename> class t,typename T0,typename T1>
  struct match<t<T0,T1>> {};

  template<typename,typename = void> struct other {};
  template struct match<other<void,void>>;
} // namespace issue2

namespace type {
  template<class T1, class T2 = float> struct A;

  template<class T3> struct B;
  template<template<class T4          > class TT1, class T5          > struct B<TT1<T5    >>   ;
  template<template<class T6, class T7> class TT2, class T8, class T9> struct B<TT2<T8, T9>> {};
  template struct B<A<int>>;
} // namespace type

namespace value {
  template<class T1, int V1 = 1> struct A;

  template<class T2> struct B;
  template<template<class T3        > class TT1, class T4        > struct B<TT1<T4    >>   ;
  template<template<class T5, int V2> class TT2, class T6, int V3> struct B<TT2<T6, V3>> {};
  template struct B<A<int>>;
} // namespace value

namespace templ {
  template <class T1> struct A;

  template<class T2, template <class T3> class T4 = A> struct B {};

  template<class T5> struct C;

  template<template<class T6> class TT1, class T7> struct C<TT1<T7>>;

  template<template<class T8, template <class T9> class> class TT2,
    class T10, template <class T11> class TT3>
  struct C<TT2<T10, TT3>> {};

  template struct C<B<int>>;
} // namespace templ

namespace class_template {
  template <class T1, class T2 = float> struct A;

  template <class T3> struct B;

  template <template <class T4> class TT1, class T5> struct B<TT1<T5>>;
  // new-note@-1 {{partial specialization matches}}

  template <class T6, class T7> struct B<A<T6, T7>> {};
  // new-note@-1 {{partial specialization matches}}

  template struct B<A<int>>;
  // new-error@-1 {{ambiguous partial specialization}}
} // namespace class_template

namespace type_pack1 {
  template<class T2> struct A;
  template<template<class ...T3s> class TT1, class T4> struct A<TT1<T4>>   ;
  template<template<class    T5 > class TT2, class T6> struct A<TT2<T6>> {};

  template<class T1> struct B;
  template struct A<B<char>>;
} // namespace type_pack1

namespace type_pack2 {
  template<class T2> struct A;
  template<template<class ...T3s> class TT1, class ...T4> struct A<TT1<T4...>>   ;
  template<template<class    T5 > class TT2, class ...T6> struct A<TT2<T6...>> {};

  template<class T1> struct B;
  template struct A<B<char>>;
} // namespace type_pack2

namespace type_pack3 {
  template<class T1, class T2 = float> struct A;

  template<class T3> struct B;

  template<template<class T4              > class TT1, class T5              > struct B<TT1<T5        >>;
  // new-note@-1 {{template is declared here}}
  template<template<class T6, class ...T7s> class TT2, class T8, class ...T9s> struct B<TT2<T8, T9s...>>;
  // old-note@-1 {{template is declared here}}

  template struct B<A<int>>;
  // expected-error@-1 {{explicit instantiation of undefined template}}
} // namespace type_pack3

namespace gcc_issue {
  template<class T1, class T2> struct A;

  template<template<class T1> class TT1, class T2> struct A<TT1<T2>, typename TT1<T2>::type>;
  // new-note@-1 {{partial specialization matches}}

  template<template<class T3, class T4> class TT2, class T5, class T6>
  struct A<TT2<T5, T6>, typename TT2<T5, T5>::type>;
  // new-note@-1 {{partial specialization matches}}
  // old-note@-2 {{template is declared here}}

  template <class T7, class T8 = T7> struct B { using type = int; };

  template struct A<B<int>, int>;
  // new-error@-1 {{ambiguous partial specializations}}
  // old-error@-2 {{explicit instantiation of undefined template}}
} // namespace gcc_issue

namespace ttp_defaults {
  template <template <class T1> class TT1> struct A {};
  // old-note@-1 2{{previous template template parameter}}

  template <template <class T2> class TT2> void f(A<TT2>);
  // new-note@-1 {{explicit instantiation candidate}}
  // old-note@-2 {{invalid explicitly-specified argument for template parameter 'TT2'}}

  // FIXME: The default arguments on the TTP are not available during partial ordering.
  template <template <class T3, class T4 = float> class TT3> void f(A<TT3>) {};
  // new-note@-1 {{explicit instantiation candidate}}
  // old-error@-2 {{template template argument has different template parameters}}
  // old-note@-3 {{too many template parameters}}

  template <class T5, class T6 = int> struct B;
  // old-note@-1 {{too many template parameters}}

  template void f<B>(A<B>);
  // new-error@-1 {{partial ordering for explicit instantiation of 'f' is ambiguous}}
  // old-error@-2 {{template template argument has different template parameters}}
  // old-error@-3 {{explicit instantiation of 'f' does not refer to a function template}}
} // namespace ttp_defaults

namespace ttp_only {
  template <template <class...    > class TT1> struct A      { static constexpr int V = 0; };
  template <template <class       > class TT2> struct A<TT2> { static constexpr int V = 1; };
  // new-note@-1 {{partial specialization matches}}
  template <template <class, class> class TT3> struct A<TT3> { static constexpr int V = 2; };
  // new-note@-1 {{partial specialization matches}}

  template <class ...          > struct B;
  template <class              > struct C;
  template <class, class       > struct D;
  template <class, class, class> struct E;

  static_assert(A<B>::V == 0); // new-error {{ambiguous partial specializations}}
  static_assert(A<C>::V == 1);
  static_assert(A<D>::V == 2);
  static_assert(A<E>::V == 0);
} // namespace ttp_only

namespace consistency {
  template<class T> struct nondeduced { using type = T; };
  template<class T8, class T9 = float> struct B;

  namespace t1 {
    template<class T1, class T2, class T3> struct A;

    template<template<class, class> class TT1,
             class T1, class T2, class T3, class T4>
    struct A<TT1<T1, T2>, TT1<T3, T4>, typename nondeduced<TT1<T1, T2>>::type> {};

    template<template<class> class UU1,
             template<class> class UU2,
             class U1, class U2>
    struct A<UU1<U1>, UU2<U2>, typename nondeduced<UU1<U1>>::type>;

    template struct A<B<int>, B<int>, B<int>>;
  } // namespace t1
  namespace t2 {
    template<class T1, class T2, class T3> struct A;

    template<template<class, class> class TT1,
             class T1, class T2, class T3, class T4>
    struct A<TT1<T1, T2>, TT1<T3, T4>, typename nondeduced<TT1<T1, T4>>::type> {};
    // new-note@-1 {{partial specialization matches}}

    template<template<class> class UU1,
             template<class> class UU2,
             class U1, class U2>
    struct A<UU1<U1>, UU2<U2>, typename nondeduced<UU1<U1>>::type>;
    // new-note@-1 {{partial specialization matches}}

    template struct A<B<int>, B<int>, B<int>>;
    // new-error@-1 {{ambiguous partial specializations}}
  } // namespace t2
  namespace t3 {
    template<class T1, class T2, class T3> struct A;

    template<template<class, class> class TT1,
             class T1, class T2, class T3, class T4>
    struct A<TT1<T1, T2>, TT1<T3, T4>, typename nondeduced<TT1<T1, T2>>::type> {};
    // new-note@-1 {{partial specialization matches}}

    template<template<class> class UU1,
             class U1, class U2>
    struct A<UU1<U1>, UU1<U2>, typename nondeduced<UU1<U1>>::type>;
    // new-note@-1 {{partial specialization matches}}

    template struct A<B<int>, B<int>, B<int>>;
    // new-error@-1 {{ambiguous partial specializations}}
  } // namespace t3
  namespace t4 {
    template<class T1, class T2, class T3> struct A;

    template<template<class, class> class TT1,
             class T1, class T2, class T3, class T4>
    struct A<TT1<T1, T2>, TT1<T3, T4>, typename nondeduced<TT1<T1, T4>>::type> {};
    // new-note@-1 {{partial specialization matches}}

    template<template<class> class UU1,
             class U1, class U2>
    struct A<UU1<U1>, UU1<U2>, typename nondeduced<UU1<U1>>::type>;
    // new-note@-1 {{partial specialization matches}}

    template struct A<B<int>, B<int>, B<int>>;
    // new-error@-1 {{ambiguous partial specializations}}
  } // namespace t4
  namespace t5 {
    template<class T1, class T2> struct A;

    template<template<class, class> class TT1,
             class T1, class T2, class T3, class T4>
    struct A<TT1<T1, T2>, TT1<T3, T4>> {};
    // new-note@-1 {{partial specialization matches}}

    template<template<class> class UU1,
             class U1, class U2>
    struct A<UU1<U1>, UU1<U2>>;
    // new-note@-1 {{partial specialization matches}}

    template struct A<B<int>, B<int>>;
    // new-error@-1 {{ambiguous partial specializations}}
  } // namespace t5
  namespace t6 {
    template<class T1, class T2> struct A;

    template<template<class, class> class TT1,
             class T1, class T2, class T3>
    struct A<TT1<T1, T2>, TT1<T1, T3>> {};
    // new-note@-1 {{partial specialization matches}}

    template<template<class> class UU1,
             class U1, class U2>
    struct A<UU1<U1>, UU1<U2>>;
    // new-note@-1 {{partial specialization matches}}

    template struct A<B<int>, B<int>>;
    // new-error@-1 {{ambiguous partial specializations}}
  } // namespace t6
} // namespace consistency

namespace classes {
  namespace canon {
    template<class T, class U> struct A {};

    template<template<class> class TT> auto f(TT<int> a) { return a; }
    // old-note@-1 2{{template template argument has different template parameters}}
    // new-note@-2 2{{substitution failure: too few template arguments}}

    A<int, float> v1;
    A<int, double> v2;

    using X = decltype(f(v1));
    // expected-error@-1 {{no matching function for call}}

    using X = decltype(f(v2));
    // expected-error@-1 {{no matching function for call}}
  } // namespace canon
  namespace expr {
    template <class T1, int E1> struct A {
      static constexpr auto val = E1;
    };
    template <template <class T3> class TT> void f(TT<int> v) {
      // old-note@-1 {{template template argument has different template parameters}}
      // new-note@-2 {{substitution failure: too few template arguments}}
      static_assert(v.val == 3);
    };
    void test() {
      f(A<int, 3>());
      // expected-error@-1 {{no matching function for call}}
    }
  } // namespace expr
  namespace packs {
    template <class T1, class ...T2s> struct A {
      static constexpr auto val = sizeof...(T2s);
    };

    template <template <class T3> class TT> void f(TT<int> v) {
      // old-note@-1 {{template template argument has different template parameters}}
      // new-note@-2 {{deduced type 'A<[...], (no argument), (no argument), (no argument)>' of 1st parameter does not match adjusted type 'A<[...], void, void, void>' of argument [with TT = A]}}
      static_assert(v.val == 3);
    };
    void test() {
      f(A<int, void, void, void>());
      // expected-error@-1 {{no matching function for call}}
    }
  } // namespace packs
} // namespace classes

namespace regression1 {
  template <typename T, typename Y> struct map {};
  template <typename T> class foo {};

  template <template <typename...> class MapType, typename Value>
  Value bar(MapType<int, Value> map);

  template <template <typename...> class MapType, typename Value>
  Value bar(MapType<int, foo<Value>> map);

  void aux() {
    map<int, foo<int>> input;
    bar(input);
  }
} // namespace regression1
