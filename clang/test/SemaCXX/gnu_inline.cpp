// RUN: %clang_cc1 -fsyntax-only -verify %s

extern inline
__attribute__((__gnu_inline__))
void gnu_inline1() {}

inline
__attribute__((__gnu_inline__)) // expected-warning {{'gnu_inline' attribute without 'extern' in C++ treated as externally available, this changed in Clang 10}}
void gnu_inline2() {}
