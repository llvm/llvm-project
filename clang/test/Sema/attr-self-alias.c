// RUN: %clang_cc1 -triple x86_64-pc-linux  -verify -emit-llvm-only %s

int self_alias(void) __attribute__((weak, alias("self_alias"))); // expected-error {{alias definition is part of a cycle}}

