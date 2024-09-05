// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

template<bool B>
struct A { };

constexpr A<false> a;
constexpr A<false> b;

constexpr int* x = nullptr;
constexpr short* y = nullptr;

namespace ExplicitArgs {
  template<typename T, typename U>
  constexpr int f(U) noexcept(noexcept(T())) {
    return 0;
  }

  template<typename T>
  constexpr int f(T*) noexcept {
    return 1;
  }

  template<>
  constexpr int f<int>(int*) noexcept {
    return 2;
  }

  static_assert(f<int>(1) == 0);
  static_assert(f<short>(y) == 1);
  static_assert(f<int>(x) == 2);

  template<typename T, typename U>
  constexpr int g(U*) noexcept(noexcept(T())) {
    return 3;
  }

  template<typename T>
  constexpr int g(T) noexcept {
    return 4;
  }

  template<>
  constexpr int g<int>(int*) noexcept {
    return 5;
  }

  static_assert(g<int>(y) == 3);
  static_assert(g<short>(1) == 4);
  static_assert(g<int>(x) == 5);
} // namespace ExplicitArgs

namespace DeducedArgs {
  template<typename T, bool B>
  constexpr int f(T, A<B>) noexcept(B) {
    return 0;
  }

  template<typename T, bool B>
  constexpr int f(T*, A<B>) noexcept(B && B) {
    return 1;
  }

  template<>
  constexpr int f(int*, A<false>) {
    return 2;
  }

  static_assert(f<int*>(x, a) == 0);
  static_assert(f<short>(y, a) == 1);
  static_assert(f<int>(x, a) == 2);
} // namespace DeducedArgs
