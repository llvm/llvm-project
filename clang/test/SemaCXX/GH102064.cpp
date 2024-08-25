// RUN: %clang_cc1 -std=c++20 -fms-extensions %s
// expected-no-diagnostics
constexpr int x = []{ __noop; return 0; }();