// RUN: %clang_cc1 -std=c++26 -verify %s

// expected-no-diagnostics

namespace GH123524 {
consteval void fn1() {}
void fn2() {
  if constexpr (&fn1 != nullptr) { }
}
}
