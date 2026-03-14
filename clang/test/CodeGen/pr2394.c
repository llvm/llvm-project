// RUN: %clang_cc1 %s -triple i386-apple-darwin10 -emit-llvm -o - | FileCheck %s
struct __attribute((packed)) x {int a : 24;};
int a(struct x* g) {
  // CHECK: load i24
  return g->a;
}
