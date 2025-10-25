// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +zicboz \
// RUN:   -Wall -Wno-unused -Werror -fsyntax-only -verify %s

#include <riscv_cmo.h>

void test(void *x) {
	__riscv_cbo_zero(x, 0); // expected-error{{too many arguments to function call, expected single argument '__x', have 2 arguments}} expected-note@riscv_cmo.h:* {{'__riscv_cbo_zero' declared here}}
	int res = __riscv_cbo_zero(x); // expected-error{{initializing 'int' with an expression of incompatible type 'void'}}
	__riscv_cbo_zero(42); // expected-error{{incompatible integer to pointer conversion passing 'int' to parameter of type 'void *'}} expected-note@riscv_cmo.h:* {{passing argument to parameter '__x' here}}
}

