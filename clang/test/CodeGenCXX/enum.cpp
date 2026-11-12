// RUN: %clang_cc1 -emit-llvm-only -verify %s
// expected-no-diagnostics

enum A { a } __attribute((packed));
int func(A x) { return x==a; }
