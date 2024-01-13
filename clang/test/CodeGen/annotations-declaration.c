// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

// Test annotation attributes are emitted for declarations.

__attribute__((annotate("bar"))) int foo();

int main() {
  return foo();
}

// CHECK: target triple
// CHECK-DAG: private unnamed_addr constant [4 x i8] c"bar\00", section "llvm.metadata"

// CHECK: @llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [{
// CHECK-SAME: { ptr @foo,
// CHECK-SAME: }], section "llvm.metadata"

