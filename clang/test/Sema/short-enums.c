// RUN: not %clang_cc1 -fsyntax-only %s -verify
// RUN: %clang_cc1 -fshort-enums -fsyntax-only %s -verify
// expected-no-diagnostics

enum x { A };
int t0[sizeof(enum x) == 1 ? 1 : -1];
