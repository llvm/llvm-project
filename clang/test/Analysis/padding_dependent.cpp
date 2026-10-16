// RUN: %clang_analyze_cc1 -std=c++20 -analyzer-checker=optin.performance -verify %s

// The static analyzer must not crash when computing the layout of a lambda
// closure type with a dependent init-capture inside a noexcept specifier.
// https://github.com/llvm/llvm-project/issues/232874
// expected-no-diagnostics
void crasher(auto&& fn) noexcept(noexcept([_ = fn] {})) {}
