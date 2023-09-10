// RUN: %clang_cc1 %s -Eonly -std=c++11 -pedantic -verify

#define2 A A

A //expected-error {{macro recursion depth limit exceeded}}

#undef A

#define2 boom(x) boom(x)

boom(5)  //expected-error {{macro recursion depth limit exceeded}}

#undef boom
