// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s

#include <x86intrin.h>

unsigned char test_addcarry_u32(unsigned char __cf, unsigned int __x,
                                unsigned int __y, unsigned int *__p) {
// CHECK-LABEL: test_addcarry_u32
// CHECK: call i8 @llvm.x86.addcarry.u32
  return _addcarry_u32(__cf, __x, __y, __p);
}

unsigned char test_addcarry_u64(unsigned char __cf, unsigned long long __x,
                                unsigned long long __y,
                                unsigned long long *__p) {
// CHECK-LABEL: test_addcarry_u64
// CHECK: call i8 @llvm.x86.addcarry.u64
  return _addcarry_u64(__cf, __x, __y, __p);
}

unsigned char test_subborrow_u32(unsigned char __cf, unsigned int __x,
                                 unsigned int __y, unsigned int *__p) {
// CHECK-LABEL: test_subborrow_u32
// CHECK: call i8 @llvm.x86.subborrow.u32
  return _subborrow_u32(__cf, __x, __y, __p);
}

unsigned char test_subborrow_u64(unsigned char __cf, unsigned long long __x,
                                 unsigned long long __y,
                                 unsigned long long *__p) {
// CHECK-LABEL: test_subborrow_u64
// CHECK: call i8 @llvm.x86.subborrow.u64
  return _subborrow_u64(__cf, __x, __y, __p);
}
