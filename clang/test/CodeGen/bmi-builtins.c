// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +bmi -emit-llvm -o - -Werror | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <x86intrin.h>

// NOTE: This should match the tests in llvm/test/CodeGen/X86/bmi-intrinsics-fast-isel.ll

// The double underscore intrinsics are for compatibility with
// AMD's BMI interface. The single underscore intrinsics
// are for compatibility with Intel's BMI interface.
// Apart from the underscores, the interfaces are identical
// except in one case: although the 'bextr' register-form
// instruction is identical in hardware, the AMD and Intel
// intrinsics are different!

unsigned short test__tzcnt_u16(unsigned short __X) {
  // CHECK-LABEL: test__tzcnt_u16
  // CHECK: zext i16 %{{.*}} to i32
  // CHECK: icmp ne i32 %{{.*}}, 0
  // CHECK: i16 @llvm.cttz.i16(i16 %{{.*}}, i1 true)
  return __tzcnt_u16(__X);
}

unsigned int test__andn_u32(unsigned int __X, unsigned int __Y) {
  // CHECK-LABEL: test__andn_u32
  // CHECK: xor i32 %{{.*}}, -1
  // CHECK: and i32 %{{.*}}, %{{.*}}
  return __andn_u32(__X, __Y);
}

unsigned int test__bextr_u32(unsigned int __X, unsigned int __Y) {
  // CHECK-LABEL: test__bextr_u32
  // CHECK: i32 @llvm.x86.bmi.bextr.32(i32 %{{.*}}, i32 %{{.*}})
  return __bextr_u32(__X, __Y);
}

unsigned int test__blsi_u32(unsigned int __X) {
  // CHECK-LABEL: test__blsi_u32
  // CHECK: sub i32 0, %{{.*}}
  // CHECK: and i32 %{{.*}}, %{{.*}}
  return __blsi_u32(__X);
}

unsigned int test__blsmsk_u32(unsigned int __X) {
  // CHECK-LABEL: test__blsmsk_u32
  // CHECK: sub i32 %{{.*}}, 1
  // CHECK: xor i32 %{{.*}}, %{{.*}}
  return __blsmsk_u32(__X);
}

unsigned int test__blsr_u32(unsigned int __X) {
  // CHECK-LABEL: test__blsr_u32
  // CHECK: sub i32 %{{.*}}, 1
  // CHECK: and i32 %{{.*}}, %{{.*}}
  return __blsr_u32(__X);
}

unsigned int test__tzcnt_u32(unsigned int __X) {
  // CHECK-LABEL: test__tzcnt_u32
  // CHECK: icmp ne i32 %{{.*}}, 0
  // CHECK: i32 @llvm.cttz.i32(i32 %{{.*}}, i1 true)
  return __tzcnt_u32(__X);
}

int test_mm_tzcnt_32(unsigned int __X) {
  // CHECK-LABEL: test_mm_tzcnt_32
  // CHECK: icmp ne i32 %{{.*}}, 0
  // CHECK: i32 @llvm.cttz.i32(i32 %{{.*}}, i1 true)
  return _mm_tzcnt_32(__X);
}

unsigned long long test__andn_u64(unsigned long __X, unsigned long __Y) {
  // CHECK-LABEL: test__andn_u64
  // CHECK: xor i64 %{{.*}}, -1
  // CHECK: and i64 %{{.*}}, %{{.*}}
  return __andn_u64(__X, __Y);
}

unsigned long long test__bextr_u64(unsigned long __X, unsigned long __Y) {
  // CHECK-LABEL: test__bextr_u64
  // CHECK: i64 @llvm.x86.bmi.bextr.64(i64 %{{.*}}, i64 %{{.*}})
  return __bextr_u64(__X, __Y);
}

unsigned long long test__blsi_u64(unsigned long long __X) {
  // CHECK-LABEL: test__blsi_u64
  // CHECK: sub i64 0, %{{.*}}
  // CHECK: and i64 %{{.*}}, %{{.*}}
  return __blsi_u64(__X);
}

unsigned long long test__blsmsk_u64(unsigned long long __X) {
  // CHECK-LABEL: test__blsmsk_u64
  // CHECK: sub i64 %{{.*}}, 1
  // CHECK: xor i64 %{{.*}}, %{{.*}}
  return __blsmsk_u64(__X);
}

unsigned long long test__blsr_u64(unsigned long long __X) {
  // CHECK-LABEL: test__blsr_u64
  // CHECK: sub i64 %{{.*}}, 1
  // CHECK: and i64 %{{.*}}, %{{.*}}
  return __blsr_u64(__X);
}

unsigned long long test__tzcnt_u64(unsigned long long __X) {
  // CHECK-LABEL: test__tzcnt_u64
  // CHECK: icmp ne i64 %{{.*}}, 0
  // CHECK: i64 @llvm.cttz.i64(i64 %{{.*}}, i1 true)
  return __tzcnt_u64(__X);
}

long long test_mm_tzcnt_64(unsigned long long __X) {
  // CHECK-LABEL: test_mm_tzcnt_64
  // CHECK: icmp ne i64 %{{.*}}, 0
  // CHECK: i64 @llvm.cttz.i64(i64 %{{.*}}, i1 true)
  return _mm_tzcnt_64(__X);
}

// Intel intrinsics

unsigned short test_tzcnt_u16(unsigned short __X) {
  // CHECK-LABEL: test_tzcnt_u16
  // CHECK: zext i16 %{{.*}} to i32
  // CHECK: icmp ne i32 %{{.*}}, 0
  // CHECK: i16 @llvm.cttz.i16(i16 %{{.*}}, i1 true)
  return _tzcnt_u16(__X);
}

unsigned int test_andn_u32(unsigned int __X, unsigned int __Y) {
  // CHECK-LABEL: test_andn_u32
  // CHECK: xor i32 %{{.*}}, -1
  // CHECK: and i32 %{{.*}}, %{{.*}}
  return _andn_u32(__X, __Y);
}

unsigned int test_bextr_u32(unsigned int __X, unsigned int __Y,
                            unsigned int __Z) {
  // CHECK-LABEL: test_bextr_u32
  // CHECK: and i32 %{{.*}}, 255
  // CHECK: and i32 %{{.*}}, 255
  // CHECK: shl i32 %{{.*}}, 8
  // CHECK: or i32 %{{.*}}, %{{.*}}
  // CHECK: i32 @llvm.x86.bmi.bextr.32(i32 %{{.*}}, i32 %{{.*}})
  return _bextr_u32(__X, __Y, __Z);
}

unsigned int test_blsi_u32(unsigned int __X) {
  // CHECK-LABEL: test_blsi_u32
  // CHECK: sub i32 0, %{{.*}}
  // CHECK: and i32 %{{.*}}, %{{.*}}
  return _blsi_u32(__X);
}

unsigned int test_blsmsk_u32(unsigned int __X) {
  // CHECK-LABEL: test_blsmsk_u32
  // CHECK: sub i32 %{{.*}}, 1
  // CHECK: xor i32 %{{.*}}, %{{.*}}
  return _blsmsk_u32(__X);
}

unsigned int test_blsr_u32(unsigned int __X) {
  // CHECK-LABEL: test_blsr_u32
  // CHECK: sub i32 %{{.*}}, 1
  // CHECK: and i32 %{{.*}}, %{{.*}}
  return _blsr_u32(__X);
}

unsigned int test_tzcnt_u32(unsigned int __X) {
  // CHECK-LABEL: test_tzcnt_u32
  // CHECK: icmp ne i32 %{{.*}}, 0
  // CHECK: i32 @llvm.cttz.i32(i32 %{{.*}}, i1 true)
  return _tzcnt_u32(__X);
}

unsigned long long test_andn_u64(unsigned long __X, unsigned long __Y) {
  // CHECK-LABEL: test_andn_u64
  // CHECK: xor i64 %{{.*}}, -1
  // CHECK: and i64 %{{.*}}, %{{.*}}
  return _andn_u64(__X, __Y);
}

unsigned long long test_bextr_u64(unsigned long __X, unsigned int __Y,
                                  unsigned int __Z) {
  // CHECK-LABEL: test_bextr_u64
  // CHECK: and i32 %{{.*}}, 255
  // CHECK: and i32 %{{.*}}, 255
  // CHECK: shl i32 %{{.*}}, 8
  // CHECK: or i32 %{{.*}}, %{{.*}}
  // CHECK: zext i32 %{{.*}} to i64
  // CHECK: i64 @llvm.x86.bmi.bextr.64(i64 %{{.*}}, i64 %{{.*}})
  return _bextr_u64(__X, __Y, __Z);
}

unsigned long long test_blsi_u64(unsigned long long __X) {
  // CHECK-LABEL: test_blsi_u64
  // CHECK: sub i64 0, %{{.*}}
  // CHECK: and i64 %{{.*}}, %{{.*}}
  return _blsi_u64(__X);
}

unsigned long long test_blsmsk_u64(unsigned long long __X) {
  // CHECK-LABEL: test_blsmsk_u64
  // CHECK: sub i64 %{{.*}}, 1
  // CHECK: xor i64 %{{.*}}, %{{.*}}
  return _blsmsk_u64(__X);
}

unsigned long long test_blsr_u64(unsigned long long __X) {
  // CHECK-LABEL: test_blsr_u64
  // CHECK: sub i64 %{{.*}}, 1
  // CHECK: and i64 %{{.*}}, %{{.*}}
  return _blsr_u64(__X);
}

unsigned long long test_tzcnt_u64(unsigned long long __X) {
  // CHECK-LABEL: test_tzcnt_u64
  // CHECK: icmp ne i64 %{{.*}}, 0
  // CHECK: i64 @llvm.cttz.i64(i64 %{{.*}}, i1 true)
  return _tzcnt_u64(__X);
}
