// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -emit-llvm -o - %s | FileCheck %s

#include <arm_acle.h>

void test_u8(unsigned char *p, unsigned char v) {
  __arm_atomic_store_with_stshh(p, v, __ATOMIC_RELAXED, 0);
}
// CHECK-LABEL: @test_u8
// CHECK: call void @llvm.aarch64.stshh(i64 0)
// CHECK-NEXT: store atomic i8 %{{.*}}, ptr %{{.*}} monotonic

void test_u16(unsigned short *p, unsigned short v) {
  __arm_atomic_store_with_stshh(p, v, __ATOMIC_RELEASE, 1);
}
// CHECK-LABEL: @test_u16
// CHECK: call void @llvm.aarch64.stshh(i64 1)
// CHECK-NEXT: store atomic i16 %{{.*}}, ptr %{{.*}} release

void test_u32(unsigned int *p, unsigned int v) {
  __arm_atomic_store_with_stshh(p, v, __ATOMIC_SEQ_CST, 0);
}
// CHECK-LABEL: @test_u32
// CHECK: call void @llvm.aarch64.stshh(i64 0)
// CHECK-NEXT: store atomic i32 %{{.*}}, ptr %{{.*}} seq_cst

void test_u64(unsigned long long *p, unsigned long long v) {
  __arm_atomic_store_with_stshh(p, v, __ATOMIC_RELAXED, 1);
}
// CHECK-LABEL: @test_u64
// CHECK: call void @llvm.aarch64.stshh(i64 1)
// CHECK-NEXT: store atomic i64 %{{.*}}, ptr %{{.*}} monotonic
