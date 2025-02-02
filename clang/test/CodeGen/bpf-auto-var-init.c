// REQUIRES: bpf-registered-target
// RUN: %clang_cc1 -triple bpf -O2 -emit-llvm %s -o - | FileCheck %s

int foo1() {
  int val;
  return val;
}
// CHECK: ret i32 0

int foo2() {
  int val[4];
  return val[2];
}
// CHECK: ret i32 0

struct val_t {
  int val;
};

int foo3() {
  struct val_t v;
  return v.val;
}
// CHECK: ret i32 0

int foo4() {
  int val = 5;
  return val;
}
// CHECK: ret i32 5
