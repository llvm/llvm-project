// RUN: %clang_cc1 -verify -std=c++1y %s

namespace PR17846 {
  template <typename T> constexpr T pi = T(3.14);
  template <typename T> constexpr T tau = 2 * pi<T>;
  constexpr double tau_double = tau<double>;
  static_assert(tau_double == 6.28, "");
}

namespace PR17848 {
  template<typename T> constexpr T var = 12345;
  template<typename T> constexpr T f() { return var<T>; }
  constexpr int k = f<int>();
  static_assert(k == 12345, "");
}

namespace NonDependent {
  template<typename T> constexpr T a = 0;
  template<typename T> constexpr T b = a<int>;
  static_assert(b<int> == 0, "");
}

namespace InstantiationDependent {
  int f(int);
  void f(char);

  template<int> constexpr int a = 1;
  template<typename T> constexpr T b = a<sizeof(sizeof(f(T())))>; // expected-error {{invalid application of 'sizeof' to an incomplete type 'void'}}

  static_assert(b<int> == 1, "");
  static_assert(b<char> == 1, ""); // expected-note {{in instantiation of}} expected-error {{not an integral constant}}

  template<typename T> void f() {
    static_assert(a<sizeof(sizeof(f(T())))> == 0, ""); // expected-error {{static assertion failed due to requirement 'a<sizeof (sizeof (f(type-parameter-0-0())))> == 0'}}
  }
}

namespace PR24483 {
  template<typename> struct A;
  template<typename... T> A<T...> models;
  template<> struct B models<>; // expected-error {{incomplete type 'struct B'}} expected-note {{forward declaration}}
}

namespace InvalidInsertPos {
  template<typename T, int N> T v;
  template<int N> decltype(v<int, N-1>) v<int, N>;
  template<> int v<int, 0>;
  int k = v<int, 500>;
}
