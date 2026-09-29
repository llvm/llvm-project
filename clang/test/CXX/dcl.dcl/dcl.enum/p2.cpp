// RUN: %clang_cc1 -std=c++11 -verify %s

enum class E : int const volatile { }; // expected-warning {{'const' and 'volatile' qualifiers in enumeration underlying type ignored}}
using T = __underlying_type(E);
using T = int;
