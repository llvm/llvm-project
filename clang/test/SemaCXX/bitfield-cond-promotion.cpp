// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -verify %s -std=c++11
// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -verify %s -std=c++14
// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -verify %s -std=c++17
// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -verify %s -std=c++20
// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -verify %s -std=c++23

void test_runtime_behavior() {
  struct {
    unsigned f : 1;
  } constexpr s{};
  
  constexpr int result = (0 ? throw 0 : s.f) - 1;
  static_assert(result == -1, "Bit-field should promote to int"); // expected-no-diagnostics
  constexpr int result2 = (1 ? s.f : s.f) - 1;
  static_assert(result2 == -1, "Bit-field should promote to int"); // expected-no-diagnostics
}
