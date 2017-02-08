// RUN: %clang_cc1 %s -triple avr-unknown-unknown -verify -fsyntax-only
struct a { int b; };

struct a test __attribute__((interrupt)); // expected-warning {{'interrupt' attribute only applies to functions}}

__attribute__((interrupt(12))) void foo(void) { } // expected-error {{'interrupt' attribute takes no arguments}}

__attribute__((interrupt)) void food() {}
