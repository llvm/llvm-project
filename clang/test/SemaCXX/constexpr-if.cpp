// RUN: %clang_cc1 -std=c++20 -verify=cxx20,expected %s
// RUN: %clang_cc1 -std=c++23 -verify=cxx23,expected %s
// RUN: %clang_cc1 -std=c++26 -verify=cxx26,expected %s

// expected-no-diagnostics

namespace GH123524 {
consteval void fn1() {}
void fn2() {
  if constexpr (&fn1 != nullptr) { }
}
}
