// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +bmi2 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -x c -ffreestanding %s -triple=i386-apple-darwin -target-feature +bmi2 -emit-llvm -o - | FileCheck %s --check-prefix=B32
// RUN: %clang_cc1 -x c++ -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +bmi2 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -x c++ -ffreestanding %s -triple=i386-apple-darwin -target-feature +bmi2 -emit-llvm -o - | FileCheck %s --check-prefix=B32


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

#ifdef __i386__
unsigned int test_mulx_u32(unsigned int __X, unsigned int __Y,
                                 unsigned int *__P) {
  // B32: mul i64
  return _mulx_u32(__X, __Y, __P);
}
#endif

#ifdef __x86_64__
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
  // CHECK: mul i128
  return _mulx_u64(__X, __Y, __P);
}
#endif

// Test constexpr handling.
#if defined(__cplusplus) && (__cplusplus >= 201103L)
char bzhi32_0[_bzhi_u32(0x89ABCDEF,   0) == 0x00000000 ? 1 : -1];
char bzhi32_1[_bzhi_u32(0x89ABCDEF,  16) == 0x0000CDEF ? 1 : -1];
char bzhi32_2[_bzhi_u32(0x89ABCDEF,  31) == 0x09ABCDEF ? 1 : -1];
char bzhi32_3[_bzhi_u32(0x89ABCDEF,  32) == 0x89ABCDEF ? 1 : -1];
char bzhi32_4[_bzhi_u32(0x89ABCDEF,  99) == 0x89ABCDEF ? 1 : -1];
char bzhi32_5[_bzhi_u32(0x89ABCDEF, 260) == 0x0000000F ? 1 : -1];

#ifdef __x86_64__
char bzhi64_0[_bzhi_u64(0x0123456789ABCDEFULL,   0) == 0x0000000000000000ULL ? 1 : -1];
char bzhi64_1[_bzhi_u64(0x0123456789ABCDEFULL,  32) == 0x0000000089ABCDEFULL ? 1 : -1];
char bzhi64_2[_bzhi_u64(0x0123456789ABCDEFULL,  99) == 0x0123456789ABCDEFULL ? 1 : -1];
char bzhi64_3[_bzhi_u64(0x0123456789ABCDEFULL, 520) == 0x00000000000000EFULL ? 1 : -1];
#endif
#endif