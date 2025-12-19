// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -fdefer-ts %s

// expected-no-diagnostics
int _Defer;
