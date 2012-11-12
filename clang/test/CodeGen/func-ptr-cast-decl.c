// RUN: %clang_cc1 -emit-llvm-only %s -verify
// expected-no-diagnostics
// PR5882

int q_sk_num(void *a);
typedef int (*fptr)(double);
void a() { ((fptr)q_sk_num)(0); }
