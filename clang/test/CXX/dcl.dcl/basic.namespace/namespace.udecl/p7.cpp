// RUN: %clang_cc1 -std=c++11 -verify %s
// RUN: %clang_cc1 -std=c++17 -verify %s
// RUN: %clang_cc1 -std=c++20 -verify %s

enum class EC { ec };
using EC::ec;
#if __cplusplus < 202002
// expected-warning@-2 {{using declaration naming a scoped enumerator is a C++20 extension}}
#else
// expected-no-diagnostics
#endif
