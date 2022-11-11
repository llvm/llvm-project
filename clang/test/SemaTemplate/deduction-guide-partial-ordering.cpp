// RUN: %clang_cc1 -std=c++20 -verify %s
// expected-no-diagnostics

namespace pr58456 {
  template<typename>
  struct s {
    constexpr s(auto) {
    }
  };

  template<typename T>
  s(T) -> s<int>;

  template<typename T> requires true
  s(T) -> s<int>;

  void f() {
    auto const y = s(0);
  }
}
