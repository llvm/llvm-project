// RUN: %clang_cc1 -triple arm64e-apple-ios -fptrauth-calls %s -verify -emit-llvm -o -
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fptrauth-calls %s -verify -emit-llvm -o -

void f(void);

// FIXME: We need a better diagnostic here.
int *pf = (int *)&f + 1; // expected-error{{cannot compile this static initializer yet}}
