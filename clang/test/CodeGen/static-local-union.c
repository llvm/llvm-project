// RUN: %clang_cc1 -emit-llvm < %s

int a(void) {static union{int a;} r[2] = {1,2};return r[1].a;}

