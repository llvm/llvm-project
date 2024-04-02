// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

__typeof_unqual__(int) u = 12;
__typeof_unqual(int) _u = 12;
