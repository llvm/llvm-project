// RUN: %clang_cc1 %s -triple avr-unknown-unknown -verify -fsyntax-only
struct a { int b; };

struct a test __attribute__((signal)); // expected-warning {{'signal' attribute only applies to functions}}

__attribute__((signal(12))) void foo(void) { } // expected-error {{'signal' attribute takes no arguments}}

__attribute__((signal)) void food(void) {}
