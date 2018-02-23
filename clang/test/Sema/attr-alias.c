// RUN: %clang_cc1 -triple x86_64-apple-darwin  -fsyntax-only -verify %s

void g() {}

// It is important that the following string be in the error message. The gcc
// testsuite looks for it to decide if a target supports aliases.

void f() __attribute__((alias("g"))); //expected-error {{only weak aliases are supported}}
