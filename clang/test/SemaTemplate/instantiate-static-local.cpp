// RUN: %clang_cc1 -std=c++2a -x c++ %s -verify

namespace use_after_instantiation {
  template<int &R> struct A { static constexpr int &value = R; };

  template<typename = void> auto S() {
    static int s;
    return A<s>{};
  }

  auto &s = decltype(S())::value;

  // This is ill-formed, but it should not crash.
  // FIXME: Right now, it does crash.
  // expected-no-diagnostics
#if 0
  template<typename = void> auto T() {
    static int s;
    struct A {
      static constexpr int &value = s; // expected-error {{static}}
    };
    return A{};
  }

  auto &t = decltype(T())::value;
#endif
}
