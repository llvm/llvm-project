// RUN: %clang_cc1 -std=c++11 -verify %s

// expected-no-diagnostics
enum class E : int const volatile { };
using T = __underlying_type(E);
using T = int;
