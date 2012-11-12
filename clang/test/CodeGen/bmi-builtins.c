// RUN: %clang_cc1 %s -O3 -triple=x86_64-apple-darwin -target-feature +bmi -emit-llvm -o - | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <x86intrin.h>

unsigned short test__tzcnt_u16(unsigned short __X) {
  // CHECK: @llvm.cttz.i16
  return __tzcnt_u16(__X);
}

unsigned int test__andn_u32(unsigned int __X, unsigned int __Y) {
  // CHECK: [[DEST:%.*]] = xor i32 %{{.*}}, -1
  // CHECK-NEXT: %{{.*}} = and i32 %{{.*}}, [[DEST]]
  return __andn_u32(__X, __Y);
}

unsigned int test__bextr_u32(unsigned int __X, unsigned int __Y) {
  // CHECK: @llvm.x86.bmi.bextr.32
  return __bextr_u32(__X, __Y);
}

unsigned int test__blsi_u32(unsigned int __X) {
  // CHECK: [[DEST:%.*]] = sub i32 0, [[SRC:%.*]]
  // CHECK-NEXT: %{{.*}} = and i32 [[SRC]], [[DEST]]
  return __blsi_u32(__X);
}

unsigned int test__blsmsk_u32(unsigned int __X) {
  // CHECK: [[DEST:%.*]] = add i32 [[SRC:%.*]], -1
  // CHECK-NEXT: %{{.*}} = xor i32 [[DEST]], [[SRC]]
  return __blsmsk_u32(__X);
}

unsigned int test__blsr_u32(unsigned int __X) {
  // CHECK: [[DEST:%.*]] = add i32 [[SRC:%.*]], -1
  // CHECK-NEXT: %{{.*}} = and i32 [[DEST]], [[SRC]]
  return __blsr_u32(__X);
}

unsigned int test_tzcnt_u32(unsigned int __X) {
  // CHECK: @llvm.cttz.i32
  return __tzcnt_u32(__X);
}

unsigned long long test__andn_u64(unsigned long __X, unsigned long __Y) {
  // CHECK: [[DEST:%.*]] = xor i64 %{{.*}}, -1
  // CHECK-NEXT: %{{.*}} = and i64 %{{.*}}, [[DEST]]
  return __andn_u64(__X, __Y);
}

unsigned long long test__bextr_u64(unsigned long __X, unsigned long __Y) {
  // CHECK: @llvm.x86.bmi.bextr.64
  return __bextr_u64(__X, __Y);
}

unsigned long long test__blsi_u64(unsigned long long __X) {
  // CHECK: [[DEST:%.*]] = sub i64 0, [[SRC:%.*]]
  // CHECK-NEXT: %{{.*}} = and i64 [[SRC]], [[DEST]]
  return __blsi_u64(__X);
}

unsigned long long test__blsmsk_u64(unsigned long long __X) {
  // CHECK: [[DEST:%.*]] = add i64 [[SRC:%.*]], -1
  // CHECK-NEXT: %{{.*}} = xor i64 [[DEST]], [[SRC]]
  return __blsmsk_u64(__X);
}

unsigned long long test__blsr_u64(unsigned long long __X) {
  // CHECK: [[DEST:%.*]] = add i64 [[SRC:%.*]], -1
  // CHECK-NEXT: %{{.*}} = and i64 [[DEST]], [[SRC]]
  return __blsr_u64(__X);
}

unsigned long long test__tzcnt_u64(unsigned long long __X) {
  // CHECK: @llvm.cttz.i64
  return __tzcnt_u64(__X);
}
