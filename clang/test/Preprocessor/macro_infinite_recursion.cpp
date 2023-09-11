// RUN: %clang_cc1 %s -Eonly -std=c++11 -pedantic -verify

#define2 A A

A //expected-error {{macro recursion depth limit exceeded}}

#undef A

#define2 A(x) A(x)

A(5)  //expected-error {{macro recursion depth limit exceeded}}

#undef A
