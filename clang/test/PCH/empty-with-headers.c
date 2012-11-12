// RUN: %clang_cc1 -fsyntax-only -std=c99 -pedantic-errors %s
// RUN: %clang_cc1 -fsyntax-only -std=c99 -emit-pch -o %t %s
// RUN: %clang_cc1 -fsyntax-only -std=c99 -pedantic-errors -include-pch %t %s

// RUN: %clang_cc1 -fsyntax-only -std=c99 -pedantic-errors -DINCLUDED %s -verify
// This last one should warn for -Wempty-translation-unit (C99 6.9p1).

#if defined(INCLUDED)

// empty except for the prefix header

#elif defined(HEADER)

typedef int my_int;
#define INCLUDED

#else

#define HEADER
#include "empty-with-headers.c"
// empty except for the header

#endif

// This should only fire if the header is not included,
// either explicitly or as a prefix header.
// expected-error{{ISO C requires a translation unit to contain at least one declaration}}
