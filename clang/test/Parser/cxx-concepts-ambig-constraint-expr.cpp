// RUN: %clang_cc1 -std=c++2a -x c++ %s -verify

// Test parsing of constraint-expressions in cases where the grammar is
// ambiguous with the expectation that the longest token sequence which matches
// the syntax is consumed without backtracking.

// type-specifier-seq in conversion-type-id
template <typename T> requires T::operator short
unsigned int foo(); // expected-error {{a type specifier is required for all declarations}}