// RUN: %clang_cc1 -O1 -triple x86_64-unknown-linux-gnu -emit-llvm -o /dev/null -verify %s
// expected-no-diagnostics

int __attribute__((const)) snprintf(char *, __SIZE_TYPE__, const char *, ...);

long foo(char c, long d) { return snprintf(&c, d, ""); }
