// RUN: %clang_cc1 -fblocks -fsyntax-only -verify %s
// expected-no-diagnostics

template<void (^)(void)> struct T;
T<nullptr> *t;
