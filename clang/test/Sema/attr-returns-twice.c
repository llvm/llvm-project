// RUN: %clang_cc1 %s -verify -fsyntax-only

int a __attribute__((returns_twice)); // expected-warning {{'returns_twice' attribute only applies to functions}}

__attribute__((returns_twice)) void t0(void) {
}

void t1(void) __attribute__((returns_twice));

void t2(void) __attribute__((returns_twice(2))); // expected-error {{'returns_twice' attribute takes no arguments}}

typedef void (*t3)(void) __attribute__((returns_twice)); // expected-warning {{'returns_twice' attribute only applies to functions}}
