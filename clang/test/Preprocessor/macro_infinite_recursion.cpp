// RUN: %clang_cc1 %s -Eonly -std=c++11 -pedantic -verify

#define A() __THIS_MACRO__()

A() //expected-error {{macro recursion depth limit exceeded}}

#undef A

#define A(x) __THIS_MACRO__(x)

A(5)  //expected-error {{macro recursion depth limit exceeded}}

#undef A
