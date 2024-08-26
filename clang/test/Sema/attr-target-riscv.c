// RUN: %clang_cc1 -triple riscv64-linux-gnu -target-feature +i -fsyntax-only -verify -std=c2x %s

//expected-note@+1 {{previous definition is here}}
int __attribute__((target("arch=rv64g"))) foo(void) { return 0; }
//expected-error@+1 {{redefinition of 'foo'}}
int __attribute__((target("arch=rv64gc"))) foo(void) { return 0; }
