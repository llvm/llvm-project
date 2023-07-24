// RUN: %clang_cc1 -triple x86_64-unknown-linux -O0 -fsanitize-cfi-cross-dso \
// RUN:     -emit-llvm -o - %s | FileCheck %s

int a;

// CHECK: define weak void @__cfi_check(i64 %[[TYPE:.*]], ptr %[[ADDR:.*]], ptr %[[DATA:.*]]) align 4096
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @__cfi_check_fail(ptr %[[DATA]], ptr %[[ADDR]])
