// RUN: %clang_cc1 -std=c++98 %s -pedantic-errors -verify=expected
// RUN: %clang_cc1 -std=c++11 %s -pedantic-errors -verify=expected
// RUN: %clang_cc1 -std=c++14 %s -pedantic-errors -verify=expected
// RUN: %clang_cc1 -std=c++17 %s -pedantic-errors -verify=expected
// RUN: %clang_cc1 -std=c++20 %s -pedantic-errors -verify=expected
// RUN: %clang_cc1 -std=c++23 %s -pedantic-errors -verify=expected
// RUN: %clang_cc1 -std=c++2c %s -pedantic-errors -verify=expected

namespace cwg3129 { // cwg3129: 3.0

float huge_f = 1e10000000000F; // expected-warning {{floating-point constant too large}}
float tiny_f = 1e-1000000000F; // expected-warning {{floating-point constant too small}}

double huge_d = 1e10000000000; // expected-warning {{floating-point constant too large}}
double tiny_d = 1e-1000000000; // expected-warning {{floating-point constant too small}}

long double huge_ld = 1e10000000000L; // expected-warning {{floating-point constant too large}}
long double tiny_ld = 1e-1000000000L; // expected-warning {{floating-point constant too small}}

} // namespace cwg3129
