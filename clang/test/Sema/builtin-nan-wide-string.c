// RUN: %clang_cc1 -fsyntax-only -verify %s

char hello = __builtin_nanf(L""); // expected-error {{incompatible pointer types passing 'int[1]' to parameter of type 'const char *'}}
