// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// PR6076
void f();
void (&g)() = (void(), f);

int a[1];
int (&b)[1] = (void(), a);
