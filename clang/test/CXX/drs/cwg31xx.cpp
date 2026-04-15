// RUN: %clang_cc1 -std=c++98 -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify=expected %s
// RUN: %clang_cc1 -std=c++11 -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify=expected %s
// RUN: %clang_cc1 -std=c++14 -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify=expected %s
// RUN: %clang_cc1 -std=c++17 -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify=expected %s
// RUN: %clang_cc1 -std=c++20 -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify=expected %s
// RUN: %clang_cc1 -std=c++23 -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify=expected %s
// RUN: %clang_cc1 -std=c++2c -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify=expected %s


// expected-no-diagnostics

namespace cwg3106 { // cwg3106: yes
#if __cplusplus >= 201102L
const char str[9] = R"(\u{1234})";
#endif
} // namespace cwg3106
