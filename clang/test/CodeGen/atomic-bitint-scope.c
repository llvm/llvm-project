// RUN: %clang_cc1 -std=c23 -triple amdgcn-amd-amdhsa -emit-llvm %s -o - | FileCheck %s

typedef unsigned _BitInt(7) U7;

// CHECK-LABEL: define {{.*}} @scoped_add7(
// CHECK: load atomic i8, ptr {{.*}} syncscope("workgroup") monotonic
// CHECK: [[OLD:%.*]] = trunc i8 {{.*}} to i7
// CHECK: cmpxchg ptr {{.*}} syncscope("workgroup") monotonic monotonic
// CHECK: [[OLD_RESULT:%.*]] = zext i7 [[OLD]] to i8
// CHECK: store i8 [[OLD_RESULT]], ptr addrspace(5) [[OLD_DEST:%.*]]
// CHECK: [[OLD_LOAD:%.*]] = load i8, ptr addrspace(5) [[OLD_DEST]]
// CHECK: [[OLD_RETURN:%.*]] = trunc i8 [[OLD_LOAD]] to i7
// CHECK: ret i7 [[OLD_RETURN]]
U7 scoped_add7(U7 *p) {
  return __scoped_atomic_fetch_add(p, (U7)1, __ATOMIC_RELAXED,
                                   __MEMORY_SCOPE_WRKGRP);
}

// CHECK-LABEL: define {{.*}} @scoped_add7_dynamic(
// CHECK: switch i32 {{.*}}, label %[[SYSTEM:[a-zA-Z0-9_.]+]] [
// CHECK: i32 1, label %[[DEVICE:[a-zA-Z0-9_.]+]]
// CHECK: i32 2, label %[[WORKGROUP:[a-zA-Z0-9_.]+]]
// CHECK: [[SYSTEM]]:
// CHECK: load atomic i8, ptr {{[^ ]+}} monotonic
// CHECK: cmpxchg ptr {{[^ ]+}}, i8 {{[^ ]+}}, i8 {{[^ ]+}} monotonic monotonic
// CHECK: [[DEVICE]]:
// CHECK: load atomic i8, ptr {{.*}} syncscope("agent") monotonic
// CHECK: cmpxchg ptr {{.*}} syncscope("agent") monotonic monotonic
// CHECK: [[WORKGROUP]]:
// CHECK: load atomic i8, ptr {{.*}} syncscope("workgroup") monotonic
// CHECK: [[NEW:%.*]] = add i7
// CHECK: cmpxchg ptr {{.*}} syncscope("workgroup") monotonic monotonic
// CHECK: [[RESULT:%.*]] = zext i7 [[NEW]] to i8
// CHECK: store i8 [[RESULT]], ptr addrspace(5) [[DEST:%.*]]
// CHECK: atomic.scope.continue:
// CHECK: [[LOADED:%.*]] = load i8, ptr addrspace(5) [[DEST]]
// CHECK: [[RETURN:%.*]] = trunc i8 [[LOADED]] to i7
// CHECK: ret i7 [[RETURN]]
U7 scoped_add7_dynamic(U7 *p, int scope) {
  return __scoped_atomic_add_fetch(p, (U7)1, __ATOMIC_RELAXED, scope);
}

typedef unsigned _BitInt(65) U65;

// CHECK-LABEL: define {{.*}} @system_add65(
// CHECK: call void @__atomic_load(i64 noundef 16,
// CHECK: call zeroext i1 @__atomic_compare_exchange(i64 noundef 16,
U65 system_add65(U65 *p) {
  return __scoped_atomic_fetch_add(p, (U65)1, __ATOMIC_RELAXED,
                                   __MEMORY_SCOPE_SYSTEM);
}
