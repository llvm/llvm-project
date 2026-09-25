// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++14 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++2c %s

// expected-no-diagnostics

namespace GH40328 {
  template <typename T, T v> struct X {};
  template <bool &v> struct X<bool &, v> {};

  template <typename T, T v> struct A { static const int k = 0; };
  template <bool &v>      struct A<bool &, v>      { static const int k = 1; };
  template <const int &v> struct A<const int &, v> { static const int k = 2; };
  template <int (&v)[3]>  struct A<int (&)[3], v>  { static const int k = 3; };
  template <void (&v)()>  struct A<void (&)(), v>  { static const int k = 4; };
  template <int *v>       struct A<int *, v>       { static const int k = 5; };

  bool b;
  extern const int ci;
  const int ci = 0;
  int arr[3];
  void fn();
  int n;

  static_assert(A<bool, true>::k == 0, "");
  static_assert(A<bool &, b>::k == 1, "");
  static_assert(A<const int &, ci>::k == 2, "");
  static_assert(A<int (&)[3], arr>::k == 3, "");
  static_assert(A<void (&)(), fn>::k == 4, "");
  static_assert(A<int *, &n>::k == 5, "");

  template <typename T, T... v> struct P { static const int k = 0; };
  template <bool &...v> struct P<bool &, v...> { static const int k = 1; };
  static_assert(P<bool &, b, b>::k == 1, "");

#if __cplusplus >= 201402L
  template <typename T, T v> const int V = 0;
  template <bool &v> const int V<bool &, v> = 1;
  static_assert(V<bool &, b> == 1, "");
#endif

  template <typename T, T v> int f(A<T, v>);
  template <bool &v> int *f(A<bool &, v>);
  int *p = f(A<bool &, b>());
} // namespace GH40328
