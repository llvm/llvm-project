// RUN: %clang_cc1 -triple=aarch64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK %s

extern void bar(void);

void foo(int *p, int *q, double *t) {
  *p = 5;
  *q = 7;
  bar();
  *t = 1.3l;
}
