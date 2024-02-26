// RUN: %clang_cc1 %s -verify -fsyntax-only

int a __attribute__((llvm_fn_attr("k", "v"))); // expected-warning {{'llvm_fn_attr' attribute only applies to functions and methods}}

void t1(void) __attribute__((llvm_fn_attr("k", "v")));

void t2(void) __attribute__((llvm_fn_attr(2, "3"))); // expected-error {{expected string literal as argument of 'llvm_fn_attr' attribute}}

void t3(void) __attribute__((llvm_fn_attr())); // expected-error {{'llvm_fn_attr' attribute requires exactly 2 arguments}}

