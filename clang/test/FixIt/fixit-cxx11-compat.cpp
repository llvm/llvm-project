// RUN: %clang_cc1 -Wc++11-compat -verify -std=c++98 %s
// RUN: cp %s %t
// RUN: not %clang_cc1 -Wc++11-compat -Werror -x c++ -std=c++98 -fixit %t
// RUN: %clang_cc1 -Wall -pedantic-errors -Wc++11-compat -Werror -x c++ -std=c++98 %t

// This is a test of the code modification hints for C++11-compatibility problems.

#define bar "bar"
const char *p = "foo"bar; // expected-warning {{will be treated as a reserved user-defined literal suffix}}
#define _bar "_bar"
const char *q = "foo"_bar; // expected-warning {{will be treated as a user-defined literal suffix}}
