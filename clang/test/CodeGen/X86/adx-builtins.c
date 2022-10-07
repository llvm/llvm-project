// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ffreestanding -target-feature +adx -emit-llvm -o - %s | FileCheck %s

#include <immintrin.h>

unsigned char test_addcarryx_u32(unsigned char __cf, unsigned int __x,
                                 unsigned int __y, unsigned int *__p) {
// CHECK-LABEL: test_addcarryx_u32
// CHECK: [[ADC:%.*]] = call { i8, i32 } @llvm.x86.addcarry.32
// CHECK: [[DATA:%.*]] = extractvalue { i8, i32 } [[ADC]], 1
// CHECK: store i32 [[DATA]], ptr %{{.*}}
// CHECK: [[CF:%.*]] = extractvalue { i8, i32 } [[ADC]], 0
  return _addcarryx_u32(__cf, __x, __y, __p);
}

unsigned char test_addcarryx_u64(unsigned char __cf, unsigned long long __x,
                                 unsigned long long __y,
                                 unsigned long long *__p) {
// CHECK-LABEL: test_addcarryx_u64
// CHECK: [[ADC:%.*]] = call { i8, i64 } @llvm.x86.addcarry.64
// CHECK: [[DATA:%.*]] = extractvalue { i8, i64 } [[ADC]], 1
// CHECK: store i64 [[DATA]], ptr %{{.*}}
// CHECK: [[CF:%.*]] = extractvalue { i8, i64 } [[ADC]], 0
  return _addcarryx_u64(__cf, __x, __y, __p);
}
