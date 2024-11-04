// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

// Test annotation attributes are still emitted when the function is used before
// it is defined with annotations.

void foo(void);
void *xxx = (void*)foo;
void __attribute__((annotate("bar"))) foo() {}

// CHECK: target triple
// CHECK-DAG: private unnamed_addr constant [4 x i8] c"bar\00", section "llvm.metadata"

// CHECK: @llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [{
// CHECK-SAME: { ptr @foo,
// CHECK-SAME: }], section "llvm.metadata"

