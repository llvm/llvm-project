// RUN: %clang_cc1 -fsyntax-only -verify=cxx23 -std=c++23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=cxx20 -std=c++20 %s
// cxx23-no-diagnostics

struct __unique {
    static constexpr auto operator()() { return 4; };  // cxx20-warning {{is a C++23 extension}}

    using P = int();
    constexpr operator P*() { return operator(); }
};

__unique four{};

int test_four() {
  // Checks that overload resolution works.
  return four();
}
