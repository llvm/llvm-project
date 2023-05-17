// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -emit-llvm -triple x86_64 -fsanitize=array-bounds %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-STRICT-0
//
// Disable checks on FAM even though the class doesn't have standard layout.

struct C {
  int head;
};

struct S : C {
  int tail[1];
};

// CHECK-LABEL: define {{.*}} @_Z8test_oneP1Si(
int test_one(S *p, int i) {
  // CHECK-STRICT-0-NOT: @__ubsan
  return p->tail[i] + (p->tail)[i];
}
