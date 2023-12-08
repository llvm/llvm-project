// RUN: %clang_cc1 -fsyntax-only -Wstrict-prototypes -verify -std=c2x %s
// RUN: %clang_cc1 -fsyntax-only -Wstrict-prototypes -verify -fno-knr-functions %s
// expected-no-diagnostics

void foo();
void bar() {}

void baz(void);
void baz() {}
