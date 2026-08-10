// RUN: %clang_cc1 -fsyntax-only -std=c99 -pedantic -W -verify %s
// RUN: %clang_cc1 -fsyntax-only -x c-header -std=c99 -pedantic-errors -W %s
// RUN: %clang_cc1 -fsyntax-only -x c++ -std=c++03 -pedantic-errors -W %s

#include "completely-empty-header-file.h"
// no-warning -- an empty file is OK

#define A_MACRO_IS_NOT_GOOD_ENOUGH 1

// In C we should get this warning, but in C++ (or a header) we shouldn't.
// expected-warning{{ISO C requires a translation unit to contain at least one declaration}}
