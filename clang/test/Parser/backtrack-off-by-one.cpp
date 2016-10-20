// RUN: %clang_cc1 -verify %s

// PR25946
// We had an off-by-one error in an assertion when annotating A<int> below.  Our
// error recovery checks if A<int> is a constructor declarator, and opens a
// TentativeParsingAction. Then we attempt to annotate the token at the exact
// position that we want to possibly backtrack to, and this used to crash.

template <typename T> class A {};

// expected-error@+1 {{expected '{' after base class list}}
template <typename T> class B : T // not ',' or '{'
// expected-error@+3 {{C++ requires a type specifier for all declarations}}
// expected-error@+2 {{expected ';' after top level declarator}}
// expected-error@+1 {{expected ';' after class}}
A<int> {
};
