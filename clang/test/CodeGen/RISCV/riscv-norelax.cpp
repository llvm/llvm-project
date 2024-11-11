// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -std=c++11 -triple riscv64 \
// RUN: -emit-llvm -o - %s | FileCheck %s

// CHECK-LABEL: define dso_local void @_Z5func1v(
// CHECK-SAME: ) #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    ret void
//
__attribute__((norelax)) void func1() {}

// CHECK-LABEL: define dso_local void @_Z5func2v(
// CHECK-SAME: ) #[[ATTR0]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    ret void
//
[[riscv::norelax]] void func2() {}

// CHECK: attributes #0 = { {{.*}} "norelax" {{.*}} }
