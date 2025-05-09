// RUN: %clang_cc1 %s -triple avr-unknown-unknown -verify -fsyntax-only
struct a { int b; };

struct a test __attribute__((interrupt)); // expected-warning {{'interrupt' attribute only applies to functions}}

__attribute__((interrupt(12))) void foo(void) { } // expected-error {{'interrupt' attribute takes no arguments}}

__attribute__((interrupt)) int fooa(void) { return 0; } // expected-warning {{'interrupt' attribute only applies to functions that have a 'void' return type}}

__attribute__((interrupt)) void foob(int a) {} // expected-warning {{'interrupt' attribute only applies to functions that have no parameters}}

__attribute__((signal)) int fooc(void) { return 0; } // expected-warning {{'signal' attribute only applies to functions that have a 'void' return type}}

__attribute__((signal)) void food(int a) {} // expected-warning {{'signal' attribute only applies to functions that have no parameters}}

__attribute__((interrupt)) void fooe(void) {}

__attribute__((interrupt)) void foof() {}

__attribute__((signal)) void foog(void) {}

__attribute__((signal)) void fooh() {}
