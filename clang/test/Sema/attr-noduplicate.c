// RUN: %clang_cc1 %s -verify -fsyntax-only

int a __attribute__((noduplicate)); // expected-warning {{'noduplicate' attribute only applies to functions}}

void t1(void) __attribute__((noduplicate));

void t2(void) __attribute__((noduplicate(2))); // expected-error {{'noduplicate' attribute takes no arguments}}

