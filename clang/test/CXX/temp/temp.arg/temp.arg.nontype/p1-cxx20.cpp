// RUN: %clang_cc1 -std=c++20 %s -verify

using size_t = __SIZE_TYPE__;

namespace CTAD {
  template<typename T> struct A { constexpr A(T) {} };
  template<A a> using DeducedA = decltype(a);

  using ATest1 = DeducedA<A(0)>;
  using ATest1 = A<int>; // expected-note {{previous}}
  using ATest1 = void; // expected-error {{different}}

  using ATest2 = DeducedA<A(0.0)>;
  using ATest2 = A<double>;

  template <size_t N> struct B {
    constexpr B(const char (&r)[N]) { __builtin_memcpy(text, r, N); }
    char text[N];
  };

  template<B b> constexpr const char *str() { return b.text; }
  static_assert(__builtin_strcmp("hello world", str<"hello world">()) == 0);
}
