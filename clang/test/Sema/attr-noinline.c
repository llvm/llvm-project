// RUN: %clang_cc1 %s -verify -fsyntax-only

int a __attribute__((noinline)); // expected-warning {{'noinline' attribute only applies to functions}}

void t1() __attribute__((noinline));

void t2() __attribute__((noinline(2))); // expected-error {{attribute takes no arguments}}

