// RUN: %clang_cc1 -emit-llvm %s -o -
// PR4556

extern void foo;
void *bar = &foo;

