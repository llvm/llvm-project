// RUN: %clang_cc1 -std=c++2a -x c++ -verify %s
// expected-no-diagnostics

template<typename T, typename U>
constexpr static bool is_same_v = false;

template<typename T>
constexpr static bool is_same_v<T, T> = true;

namespace PR56154 {
  template <int N> concept C0 = (N == 0);
  template <int N, int N2> concept C0x = C0<N>;
  template <int N1, int N2> concept C00 = C0x<N1, N2> && C0<N2>;

  template<int N1, int N2>
  struct A {
    void f() requires C00<N1, N2>;
    void f() requires C0x<N1, N2> = delete;

    static short g() requires C00<N1, N2>;
    static int g() requires C0x<N1, N2>;
  };
  void h(A<0, 0> a) {
    a.f();
    static_assert(is_same_v<decltype(&A<0, 0>::g), short(*)()>);
  }
}
