// REQUIRES: bpf-registered-target
// RUN: %clang_cc1 -no-opaque-pointers -triple bpf -emit-llvm -target-feature +alu32 %s -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -triple bpf -emit-llvm -target-cpu v3 %s -o - | FileCheck %s

void test_generic_constraints(int var32, long var64) {
  asm("%0 = %1"
      : "=r"(var32)
      : "0"(var32));
  // CHECK: [[R32_ARG:%[a-zA-Z0-9]+]] = load i32, i32*
  // CHECK: call i32 asm "$0 = $1", "=r,0"(i32 [[R32_ARG]])

  asm("%0 = %1"
      : "=r"(var64)
      : "0"(var64));
  // CHECK: [[R64_ARG:%[a-zA-Z0-9]+]] = load i64, i64*
  // CHECK: call i64 asm "$0 = $1", "=r,0"(i64 [[R64_ARG]])

  asm("%0 = %1"
      : "=r"(var64)
      : "r"(var64));
  // CHECK: [[R64_ARG:%[a-zA-Z0-9]+]] = load i64, i64*
  // CHECK: call i64 asm "$0 = $1", "=r,r"(i64 [[R64_ARG]])
}

void test_constraint_w(int a) {
  asm("%0 = %1"
      : "=w"(a)
      : "w"(a));
  // CHECK: [[R32_ARG:%[a-zA-Z0-9]+]] = load i32, i32*
  // CHECK: call i32 asm "$0 = $1", "=w,w"(i32 [[R32_ARG]])
}
