// RUN: %clang_cc1 -std=c++17 -verify %s

template<typename T> struct A {
  template<typename U> struct B {
    B(...);
    B(const B &) = default;
  };
  template<typename U> B(U) -> B<U>;
};
A<void>::B b = 123;
A<void>::B copy = b;

using T = decltype(b);
using T = A<void>::B<int>;

using Copy = decltype(copy);
using Copy = A<void>::B<int>;

namespace GH94614 {

template <class, class> struct S {};

struct trouble_1 {
} constexpr t1;
struct trouble_2 {
} constexpr t2;
struct trouble_3 {
} constexpr t3;
struct trouble_4 {
} constexpr t4;
struct trouble_5 {
} constexpr t5;
struct trouble_6 {
} constexpr t6;
struct trouble_7 {
} constexpr t7;
struct trouble_8 {
} constexpr t8;
struct trouble_9 {
} constexpr t9;

template <class U, class... T> struct Unrelated {
  using Trouble = S<U, T...>;

  template <class... V> using Trouble2 = S<V..., T...>;
};

template <class T, class U> struct Outer {
  using Trouble = S<U, T>;

  template <class V> using Trouble2 = S<V, T>;

  template <class V> using Trouble3 = S<U, T>;

  template <class V> struct Inner {
    template <class W> struct Paranoid {
      using Trouble4 = S<W, T>;

      template <class... X> using Trouble5 = S<X..., T>;
    };

    Inner(trouble_1, V v, Trouble trouble) {}
    Inner(trouble_2, V v, Trouble2<V> trouble) {}
    Inner(trouble_3, V v, Trouble3<V> trouble) {}
    Inner(trouble_4, V v, typename Unrelated<U, T>::template Trouble2<V> trouble) {}
    Inner(trouble_5, V v, typename Unrelated<U, T>::Trouble trouble) {}
    Inner(trouble_6, V v, typename Unrelated<V, T>::Trouble trouble) {}
    Inner(trouble_7, V v, typename Paranoid<V>::Trouble4 trouble) {}
    Inner(trouble_8, V v, typename Paranoid<V>::template Trouble5<V> trouble) {}
    template <class W>
    Inner(trouble_9, V v, W w, typename Paranoid<V>::template Trouble5<W> trouble) {}
  };
};

S<int, char> s;

Outer<char, int>::Inner _1(t1, 42, s);
Outer<char, int>::Inner _2(t2, 42, s);
Outer<char, int>::Inner _3(t3, 42, s);
Outer<char, int>::Inner _4(t4, 42, s);
Outer<char, int>::Inner _5(t5, 42, s);
Outer<char, int>::Inner _6(t6, 42, s);
Outer<char, int>::Inner _7(t7, 42, s);
Outer<char, int>::Inner _8(t8, 42, s);
Outer<char, int>::Inner _9(t9, 42, 24, s);

// Make sure we don't accidentally inject the TypedefNameDecl into the TU.
Trouble should_not_be_in_the_tu_decl; // expected-error {{unknown type name 'Trouble'}}

} // namespace GH94614
