// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s
// PR1352

struct foo {
  int x;
};

void copy(volatile struct foo *p, struct foo *q) {
  // CHECK: call void @llvm.memcpy
  *p = *q;
}
