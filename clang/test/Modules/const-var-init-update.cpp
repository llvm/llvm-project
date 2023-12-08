// RUN: %clang_cc1 -std=c++1z -fmodules %s -verify
// expected-no-diagnostics

#pragma clang module build std
module std { module limits {} module other {} }
#pragma clang module contents
#pragma clang module begin std.limits
template<typename T> struct numeric_limits {
  static constexpr T __max = 5;
  static constexpr T max() { return __max; }
};
#pragma clang module end
#pragma clang module begin std.other
inline void f() { numeric_limits<int> nl; }
#pragma clang module end
#pragma clang module endbuild

#pragma clang module build module_b
module module_b {}
#pragma clang module contents
#pragma clang module begin module_b
#pragma clang module import std.limits
constexpr int a = numeric_limits<int>::max();
#pragma clang module end
#pragma clang module endbuild

#pragma clang module import std.limits
#pragma clang module import module_b
constexpr int b = a;
static_assert(b == 5);
