// RUN: %clang_cc1 %s -verify -fsyntax-only

int a __attribute__((noinline)); // expected-warning {{'noinline' attribute only applies to functions and statements}}

void t1(void) __attribute__((noinline));

void t2(void) __attribute__((noinline(2))); // expected-error {{'noinline' attribute takes no arguments}}

