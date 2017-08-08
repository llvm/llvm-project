// RUN: %clang_cc1 %s -verify -fsyntax-only -fms-extensions

int __unaligned * p1; // expected-note {{previous definition is here}}
int * p1; // expected-error {{redefinition of 'p1' with a different type: 'int *' vs '__unaligned int *'}}
