// RUN: %clang_cc1 -fsyntax-only -fms-extensions -verify %s
// expected-no-diagnostics

#define inline _inline
#undef  inline

int x;
