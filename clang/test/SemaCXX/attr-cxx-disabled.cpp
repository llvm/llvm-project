// RUN: %clang_cc1 -fsyntax-only -fno-double-square-bracket-attributes -verify -pedantic -std=c++11 -DERRORS %s
// RUN: %clang_cc1 -fsyntax-only -fdouble-square-bracket-attributes -verify -pedantic -std=c++11 %s

struct [[]] S {};

#ifdef ERRORS
// expected-error@-3 {{declaration of anonymous struct must be a definition}}
// expected-warning@-4 {{declaration does not declare anything}}
#else
// expected-no-diagnostics
#endif

