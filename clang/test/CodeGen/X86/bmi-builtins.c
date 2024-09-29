// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +bmi -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,TZCNT
// RUN: %clang_cc1 -x c -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 -ffreestanding %s -triple=x86_64-windows-msvc -emit-llvm -o - -Wall -Werror -DTEST_TZCNT | FileCheck %s --check-prefix=TZCNT
// RUN: %clang_cc1 -x c++ -std=c++11 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +bmi -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,TZCNT
// RUN: %clang_cc1 -x c++ -std=c++11 -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 -ffreestanding %s -triple=x86_64-windows-msvc -emit-llvm -o - -Wall -Werror -DTEST_TZCNT | FileCheck %s --check-prefix=TZCNT


#include <immintrin.h>

// NOTE: This should match the tests in llvm/test/CodeGen/X86/bmi-intrinsics-fast-isel.ll

// The double underscore intrinsics are for compatibility with
// AMD's BMI interface. The single underscore intrinsics
// are for compatibility with Intel's BMI interface.
// Apart from the underscores, the interfaces are identical
// except in one case: although the 'bextr' register-form
// instruction is identical in hardware, the AMD and Intel
// intrinsics are different!

unsigned short test_tzcnt_u16(unsigned short __X) {
// TZCNT-LABEL: test_tzcnt_u16
// TZCNT: i16 @llvm.cttz.i16(i16 %{{.*}}, i1 false)
  return _tzcnt_u16(__X);
}

unsigned short test__tzcnt_u16(unsigned short __X) {
// TZCNT-LABEL: test__tzcnt_u16
// TZCNT: i16 @llvm.cttz.i16(i16 %{{.*}}, i1 false)
  return __tzcnt_u16(__X);
}

unsigned int test__tzcnt_u32(unsigned int __X) {
// TZCNT-LABEL: test__tzcnt_u32
// TZCNT: i32 @llvm.cttz.i32(i32 %{{.*}}, i1 false)
  return __tzcnt_u32(__X);
}

int test_mm_tzcnt_32(unsigned int __X) {
// TZCNT-LABEL: test_mm_tzcnt_32
// TZCNT: i32 @llvm.cttz.i32(i32 %{{.*}}, i1 false)
  return _mm_tzcnt_32(__X);
}

unsigned int test_tzcnt_u32(unsigned int __X) {
// TZCNT-LABEL: test_tzcnt_u32
// TZCNT: i32 @llvm.cttz.i32(i32 %{{.*}}, i1 false)
  return _tzcnt_u32(__X);
}

#ifdef __x86_64__
unsigned long long test__tzcnt_u64(unsigned long long __X) {
// TZCNT-LABEL: test__tzcnt_u64
// TZCNT: i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
  return __tzcnt_u64(__X);
}

long long test_mm_tzcnt_64(unsigned long long __X) {
// TZCNT-LABEL: test_mm_tzcnt_64
// TZCNT: i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
  return _mm_tzcnt_64(__X);
}

unsigned long long test_tzcnt_u64(unsigned long long __X) {
// TZCNT-LABEL: test_tzcnt_u64
// TZCNT: i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
  return _tzcnt_u64(__X);
}
#endif

#if !defined(TEST_TZCNT)
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

#ifdef __x86_64__
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
#endif

// Intel intrinsics

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

unsigned int test_bextr2_u32(unsigned int __X, unsigned int __Y) {
// CHECK-LABEL: test_bextr2_u32
// CHECK: i32 @llvm.x86.bmi.bextr.32(i32 %{{.*}}, i32 %{{.*}})
  return _bextr2_u32(__X, __Y);
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

#ifdef __x86_64__
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

unsigned long long test_bextr2_u64(unsigned long long __X,
                                   unsigned long long __Y) {
// CHECK-LABEL: test_bextr2_u64
// CHECK: i64 @llvm.x86.bmi.bextr.64(i64 %{{.*}}, i64 %{{.*}})
  return _bextr2_u64(__X, __Y);
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
#endif

#endif // !defined(TEST_TZCNT)

// Test constexpr handling.
#if defined(__cplusplus) && (__cplusplus >= 201103L)
char bextr32_0[__bextr_u32(0x00000000, 0x00000000) == 0x00000000 ? 1 : -1];
char bextr32_1[__bextr_u32(0x000003F0, 0xFFFF1004) == 0x0000003F ? 1 : -1];
char bextr32_2[__bextr_u32(0x000003F0, 0xFFFF3008) == 0x00000003 ? 1 : -1];

char bextr32_3[_bextr2_u32(0x00000000, 0x00000000) == 0x00000000 ? 1 : -1];
char bextr32_4[_bextr2_u32(0x000003F0, 0xFFFF1004) == 0x0000003F ? 1 : -1];
char bextr32_5[_bextr2_u32(0x000003F0, 0xFFFF3008) == 0x00000003 ? 1 : -1];

char bextr32_6[_bextr_u32(0x00000000, 0x00000000, 0x00000000) == 0x00000000 ? 1 : -1];
char bextr32_7[_bextr_u32(0x000003F0, 0xFFFFFF04, 0xFFFFFF10) == 0x0000003F ? 1 : -1];
char bextr32_8[_bextr_u32(0x000003F0, 0xFFFFFF08, 0xFFFFFF30) == 0x00000003 ? 1 : -1];

#ifdef __x86_64__
char bextr64_0[__bextr_u64(0x0000000000000000ULL, 0x0000000000000000ULL) == 0x0000000000000000ULL ? 1 : -1];
char bextr64_1[__bextr_u64(0xF000000000000001ULL, 0x0000000000004001ULL) == 0x7800000000000000ULL ? 1 : -1];
char bextr64_2[__bextr_u64(0xF000000000000001ULL, 0xFFFFFFFFFFFF1001ULL) == 0x0000000000000000ULL ? 1 : -1];

char bextr64_3[_bextr2_u64(0x0000000000000000ULL, 0x0000000000000000ULL) == 0x0000000000000000ULL ? 1 : -1];
char bextr64_4[_bextr2_u64(0xF000000000000001ULL, 0x0000000000004001ULL) == 0x7800000000000000ULL ? 1 : -1];
char bextr64_5[_bextr2_u64(0xF000000000000001ULL, 0xFFFFFFFFFFFF1001ULL) == 0x0000000000000000ULL ? 1 : -1];

char bextr64_6[_bextr_u64(0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL) == 0x0000000000000000ULL ? 1 : -1];
char bextr64_7[_bextr_u64(0xF000000000000001ULL, 0x0000000000000001ULL, 0x0000000000000040ULL) == 0x7800000000000000ULL ? 1 : -1];
char bextr64_8[_bextr_u64(0xF000000000000001ULL, 0xFFFFFFFFFFFFFF01ULL, 0xFFFFFFFFFFFFFF10ULL) == 0x0000000000000000ULL ? 1 : -1];
#endif
#endif