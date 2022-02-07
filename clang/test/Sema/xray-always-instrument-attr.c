// RUN: %clang_cc1 %s -verify -fsyntax-only -std=c11
void foo(void) __attribute__((xray_always_instrument));

struct __attribute__((xray_always_instrument)) a { int x; }; // expected-warning {{'xray_always_instrument' attribute only applies to functions and Objective-C methods}}

void bar(void) __attribute__((xray_always_instrument("not-supported"))); // expected-error {{'xray_always_instrument' attribute takes no arguments}}
