// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 \
// RUN: -emit-llvm -o - %s | FileCheck %s

// CHECK-LABEL: define dso_local void @func1(
// CHECK-SAME: ) #0 {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    ret void
//
__attribute__((norelax)) void func1() {}

// CHECK-LABEL: define dso_local void @func2(
// CHECK-SAME: ) #0 {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    ret void
//
[[riscv::norelax]] void func2() {}

// CHECK: attributes #0 = { {{.*}} "norelax" {{.*}} }
