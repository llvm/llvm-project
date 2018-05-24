// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +bmi2 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -ffreestanding %s -triple=i386-apple-darwin -target-feature +bmi2 -emit-llvm -o - | FileCheck %s --check-prefix=B32


#include <immintrin.h>

unsigned int test_bzhi_u32(unsigned int __X, unsigned int __Y) {
  // CHECK: @llvm.x86.bmi.bzhi.32
  return _bzhi_u32(__X, __Y);
}

unsigned int test_pdep_u32(unsigned int __X, unsigned int __Y) {
  // CHECK: @llvm.x86.bmi.pdep.32
  return _pdep_u32(__X, __Y);
}

unsigned int test_pext_u32(unsigned int __X, unsigned int __Y) {
  // CHECK: @llvm.x86.bmi.pext.32
  return _pext_u32(__X, __Y);
}

unsigned int test_mulx_u32(unsigned int __X, unsigned int __Y,
                                 unsigned int *__P) {
  // CHECK: @test_mulx_u32
  // CHECK-NOT: mul i64
  // B32: @test_mulx_u32
  // B32: mul i64
  return _mulx_u32(__X, __Y, __P);
}

unsigned long long test_bzhi_u64(unsigned long long __X, unsigned long long __Y) {
  // CHECK: @llvm.x86.bmi.bzhi.64
  return _bzhi_u64(__X, __Y);
}

unsigned long long test_pdep_u64(unsigned long long __X, unsigned long long __Y) {
  // CHECK: @llvm.x86.bmi.pdep.64
  return _pdep_u64(__X, __Y);
}

unsigned long long test_pext_u64(unsigned long long __X, unsigned long long __Y) {
  // CHECK: @llvm.x86.bmi.pext.64
  return _pext_u64(__X, __Y);
}

unsigned long long test_mulx_u64(unsigned long long __X, unsigned long long __Y,
                                 unsigned long long *__P) {
  // CHECK: @test_mulx_u64
  // CHECK: mul i128
  return _mulx_u64(__X, __Y, __P);
}
