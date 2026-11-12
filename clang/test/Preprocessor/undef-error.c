// RUN: %clang_cc1 %s -pedantic-errors -Wno-empty-translation-unit -verify 
// PR2045

#define b
/* expected-error {{extra tokens at end of #undef directive}} */ #undef a b
