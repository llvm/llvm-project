// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ffreestanding -target-feature +adx -emit-llvm -o - %s | FileCheck %s

#include <x86intrin.h>

unsigned char test_addcarryx_u32(unsigned char __cf, unsigned int __x,
                                 unsigned int __y, unsigned int *__p) {
// CHECK-LABEL: test_addcarryx_u32
// CHECK: call i8 @llvm.x86.addcarryx.u32
  return _addcarryx_u32(__cf, __x, __y, __p);
}

unsigned char test_addcarryx_u64(unsigned char __cf, unsigned long long __x,
                                 unsigned long long __y,
                                 unsigned long long *__p) {
// CHECK-LABEL: test_addcarryx_u64
// CHECK: call i8 @llvm.x86.addcarryx.u64
  return _addcarryx_u64(__cf, __x, __y, __p);
}
