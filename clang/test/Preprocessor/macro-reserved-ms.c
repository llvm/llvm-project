// RUN: %clang_cc1 -fsyntax-only -fms-extensions -verify %s

#define inline _inline
#undef  inline // expected-warning {{keyword or identifier with special meaning is used as a macro name}}

int x;
