// RUN: %clang_cc1 -std=c++98 -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify %s
// RUN: %clang_cc1 -std=c++11 -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify %s
// RUN: %clang_cc1 -std=c++14 -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify %s
// RUN: %clang_cc1 -std=c++17 -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify %s
// RUN: %clang_cc1 -std=c++20 -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify %s
// RUN: %clang_cc1 -std=c++23 -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify %s
// RUN: %clang_cc1 -std=c++2c -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify %s

// cwg3015: partial

// An intervening empty macro prevents header-name tokenization.
#define EMPTY
#include EMPTY "name with \"quotes\".h"
// expected-error@-1 {{'name with \"quotes\".h' file not found}}
