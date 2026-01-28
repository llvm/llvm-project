// RUN: %clang_cc1 -E -pedantic-errors %s -verify -std=c++98
// RUN: %clang_cc1 -E -pedantic-errors %s -verify -std=c++11
// RUN: %clang_cc1 -E -pedantic-errors %s -verify -std=c++14
// RUN: %clang_cc1 -E -pedantic-errors %s -verify -std=c++17
// RUN: %clang_cc1 -E -pedantic-errors %s -verify -std=c++20
// RUN: %clang_cc1 -E -pedantic-errors %s -verify=expected,since-cxx23 -std=c++23 -Wno-c23-extensions
// RUN: %clang_cc1 -E -pedantic-errors %s -verify=expected,since-cxx23 -std=c++2c -Wno-c23-extensions

// Test 1: Top-level comma
// expected-error@+1 {{expected end of line in preprocessor expression}}
#if 1, 2
#endif

// Test 2: Comma in conditional expression(CWG3017)
// Per CWG 3017, this exact case highlights the specification gap
// where C++ lacks explicit prohibition of comma operators in #if
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

// Test 5: Comma in #elif
#if 0
#elif (1, 2) // expected-error {{comma operator in operand of #if}}
#endif

// Test 6: Leading comma (syntax error)
// expected-error@+1 {{invalid token at start of a preprocessor expression}}
#if ,
#endif

// Test 7: Comma in #embed limit parameter (C++23+)
#if __cplusplus >= 202302L
// since-cxx23-error@+1 {{expected ')'}}
#embed "jk.txt" limit(1, 2)
#endif
