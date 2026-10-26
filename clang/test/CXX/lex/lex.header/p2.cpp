// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s

#if __has_include("\")
#endif

#include "\" // expected-error {{'\' file not found}}
