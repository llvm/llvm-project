// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu_ilp32 -emit-llvm -o - %s | FileCheck %s --check-prefixes=LE,CHECK
// RUN: %clang_cc1 -triple aarch64_be-unknown-linux-gnu_ilp32 -emit-llvm -o - %s | FileCheck %s --check-prefixes=BE,CHECK

// LE: target datalayout = "e-m:e-p:32:32-
// BE: target datalayout = "E-m:e-p:32:32-

unsigned long pointer_size(void) { return sizeof(void *); }

// CHECK-LABEL: define{{.*}} i32 @pointer_size(
// CHECK: ret i32 4
