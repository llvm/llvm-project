// RUN: %clang_cc1 -fsyntax-only -std=c++17 %s -verify
// expected-no-diagnostics

void test_noexcept() {
  const auto lambda = [](int x) { return x + 1; };
  static_assert(noexcept((int (*)(int))(lambda)),
                "Lambda-to-function-pointer conversion is expected to be noexcept");
}
