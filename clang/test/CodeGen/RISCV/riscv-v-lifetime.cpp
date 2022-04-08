// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -no-opaque-pointers -std=c++11 -triple riscv64 -target-feature +v \
// RUN:   -O1 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

#include <riscv_vector.h>

vint32m1_t Baz();

// CHECK-LABEL: @_Z4Testv(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[A:%.*]] = alloca <vscale x 2 x i32>*, align 8
// CHECK-NEXT:    [[REF_TMP:%.*]] = alloca <vscale x 2 x i32>, align 4
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <vscale x 2 x i32>** [[A]] to i8*
// CHECK-NEXT:    call void @llvm.lifetime.start.p0i8(i64 8, i8* [[TMP0]]) #[[ATTR3:[0-9]+]]
// CHECK-NEXT:    [[TMP1:%.*]] = bitcast <vscale x 2 x i32>* [[REF_TMP]] to i8*
// CHECK-NEXT:    call void @llvm.lifetime.start.p0i8(i64 -1, i8* [[TMP1]]) #[[ATTR3]]
// CHECK:         [[TMP4:%.*]] = bitcast <vscale x 2 x i32>* [[REF_TMP]] to i8*
// CHECK-NEXT:    call void @llvm.lifetime.end.p0i8(i64 -1, i8* [[TMP4]]) #[[ATTR3]]
// CHECK-NEXT:    [[TMP5:%.*]] = bitcast <vscale x 2 x i32>** [[A]] to i8*
// CHECK-NEXT:    call void @llvm.lifetime.end.p0i8(i64 8, i8* [[TMP5]]) #[[ATTR3]]
//
vint32m1_t Test() {
  const vint32m1_t &a = Baz();
  return a;
}
