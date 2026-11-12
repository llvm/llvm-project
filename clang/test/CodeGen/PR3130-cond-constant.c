// RUN: %clang_cc1 -emit-llvm %s -o -

int a = 2.0 ? 1 : 2;
