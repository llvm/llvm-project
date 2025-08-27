// RUN: %clang_cc1 -E -pedantic-errors %s -verify -std=c++98
// RUN: %clang_cc1 -E -pedantic-errors %s -verify -std=c++11
// RUN: %clang_cc1 -E -pedantic-errors %s -verify -std=c++14
// RUN: %clang_cc1 -E -pedantic-errors %s -verify -std=c++17
// RUN: %clang_cc1 -E -pedantic-errors %s -verify -std=c++20
// RUN: %clang_cc1 -E -pedantic-errors %s -verify -std=c++23

// Test 1: Top-level comma
// expected-error@+1 {{expected end of line in preprocessor expression}}
#if 1, 2
#endif

// Test 2: Comma in conditional expression
// expected-error@+1 {{comma operator in operand of #if}}
#if 1 ? 1, 0 : 3
#endif

// Test 3: Parenthesized comma
// expected-error@+1 {{comma operator in operand of #if}}
#if (1, 2)
#endif

// Test 4: Multiple commas
// expected-error@+1 {{expected end of line in preprocessor expression}}
#if 1, 2, 3
#endif
