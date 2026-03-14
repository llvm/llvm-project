// RUN: %clang_cc1 -emit-llvm < %s | grep 'extern_weak' | count 1

struct S;
void __attribute__((weak)) foo1(struct S);
void (*foo2)(struct S) = foo1;
