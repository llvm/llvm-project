// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -std=c++11 -triple riscv64 -target-feature +v \
// RUN:   -O1 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

#include <riscv_vector.h>

vint32m1_t Baz();

// CHECK-LABEL: @_Z4Testv(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[A:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[REF_TMP:%.*]] = alloca <vscale x 2 x i32>, align 4
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 8, ptr [[A]]) #[[ATTR3:[0-9]+]]
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 -1, ptr [[REF_TMP]]) #[[ATTR3]]
// CHECK:    call void @llvm.lifetime.end.p0(i64 -1, ptr [[REF_TMP]]) #[[ATTR3]]
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 8, ptr [[A]]) #[[ATTR3]]
//
vint32m1_t Test() {
  const vint32m1_t &a = Baz();
  return a;
}
