// RUN: %clang_cc1 -triple arm64e-apple-ios -fptrauth-calls %s -verify -emit-llvm

void f(void);

int *pf = (int *)&f + 1; // expected-error{{cannot compile this static initializer yet}}
