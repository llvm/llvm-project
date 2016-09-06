// RUN: %clang_cc1 -std=c++11 -verify %s

enum class EC { ec };
using EC::ec; // expected-error {{using declaration cannot refer to a scoped enumerator}}
