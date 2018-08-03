// RUN: %clang_cc1 -fsyntax-only -verify %s

void __attribute__((disable_tail_calls,naked)) foo1(int a) { // expected-error {{'naked' and 'disable_tail_calls' attributes are not compatible}} expected-note {{conflicting attribute is here}}
  __asm__("");
}

void __attribute__((naked,disable_tail_calls)) foo2(int a) { // expected-error {{'disable_tail_calls' and 'naked' attributes are not compatible}} expected-note {{conflicting attribute is here}}
  __asm__("");
}

int g0 __attribute__((disable_tail_calls)); // expected-warning {{'disable_tail_calls' attribute only applies to functions and Objective-C methods}}

int foo3(int a) __attribute__((disable_tail_calls("abc"))); // expected-error {{'disable_tail_calls' attribute takes no arguments}}
