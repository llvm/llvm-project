// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s
struct __attribute((packed)) x {int a : 24;};
int a(struct x* g) {
  // CHECK: load i16
  // CHECK: load i8
  return g->a;
}
