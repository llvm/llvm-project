// RUN: %clang_cc1 -fsyntax-only -std=c++17  -verify %s
// expected-no-diagnostics
template <typename a, int* = nullptr>
struct e {
    e(a) {}
};
e c(0);
