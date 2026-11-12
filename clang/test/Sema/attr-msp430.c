// RUN: %clang_cc1 -triple msp430-unknown-unknown -fsyntax-only -verify %s

__attribute__((interrupt(1))) int t; // expected-warning {{'interrupt' attribute only applies to functions}}

int i;
__attribute__((interrupt(i))) void f(void); // expected-error {{'interrupt' attribute requires an integer constant}}
__attribute__((interrupt(1, 2))) void f2(void); // expected-error {{'interrupt' attribute takes one argument}}
__attribute__((interrupt(1))) int f3(void); // expected-warning {{MSP430 'interrupt' attribute only applies to functions that have a 'void' return type}}
__attribute__((interrupt(1))) void f4(int a); // expected-warning {{MSP430 'interrupt' attribute only applies to functions that have no parameters}}
__attribute__((interrupt(64))) void f5(void); // expected-error {{'interrupt' attribute parameter 64 is out of bounds}}

__attribute__((interrupt(0))) void f6(void);
__attribute__((interrupt(63))) void f7(void);
