// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify %s

//expected-warning@+1 {{unknown attribute 'target_version' ignored}}
int __attribute__((target_version("aes"))) foo(void) { return 3; }
