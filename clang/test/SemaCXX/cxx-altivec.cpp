// RUN: %clang_cc1 -triple=powerpc-apple-darwin8 -target-feature +altivec -fsyntax-only -verify %s

struct Vector {
	__vector float xyzw;
} __attribute__((vecreturn)) __attribute__((vecreturn));  // expected-error {{'vecreturn' attribute cannot be repeated}}
