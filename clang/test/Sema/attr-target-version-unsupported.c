// RUN: %clang_cc1 -triple x86_64-unknown-unknown  -fsyntax-only -verify %s

//expected-error@+1 {{target_version attribute is not supported on this target}}
int __attribute__((target_version("aes"))) foo(void) { return 3; }
