// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +bmi -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,TZCNT
// RUN: %clang_cc1 -x c++ -std=c++11 -std -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +bmi -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,TZCNT
// RUN: %clang_cc1 -x c -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 -ffreestanding %s -triple=x86_64-windows-msvc -emit-llvm -o - -Wall -Werror -DTEST_TZCNT | FileCheck %s --check-prefix=TZCNT
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

char cttz16_0[__tzcnt_u16(0x0000) == 16 ? 1 : -1];
char cttz16_1[__tzcnt_u16(0x10F0) == 4 ? 1 : -1];

char cttz32_0[__tzcnt_u32(0x0FF00000) == 20 ? 1 : -1];
char cttz32_1[__tzcnt_u32(0x000000F0) == 4 ? 1 : -1];

char mm_cttz32_0[_mm_tzcnt_u32(0x00030F00) == 8 ? 1 : -1];
char mm_cttz32_1[_mm_tzcnt_u32(0x01000000) == 24 ? 1 : -1];

//Intel
char _tzcnt_u16_0[_tzcnt_u16(0x0010) == 4 ? 1 : -1];
char _tzcnt_u16_1[_tzcnt_u16(0x0100) == 8 ? 1 : -1];

char _tzcnt_u32_0[_tzcnt_u32(0x00110011) == 0 ? 1 : -1;
char _tzcnt_u32_1[_tzcnt_u32(0x10011000) == 12 ? 1 : -1;

#ifdef __x86_64__
char mm_cttz64_0[mm_tzcnt_u64(0x0000000000000000ULL) == 0 ? 1 : -1];
char mm_cttz64_1[mm_tzcnt_u64(0xF000010000000000ULL) == 40 ? 1 : -1];

//Intel
char __tzcnt_u64_0[__tzcnt_u64(0x0100000000000000ULL) == 56 ? 1 : -1];
char __tzcnt_u64_1[__tzcnt_u64(0xF000000000000001ULL) == 0 ? 1 : -1];
#endif //__X86_64__

#if !defined(TEST_TZCNT)

//ANDN
char andn_u32_0[__andn_u32(0X0FFF0FFF, 0XFFFFFFFF) == 0xF000F000 ? 1 : -1];
char andn_u32_1[__andn_u32(0x0F0FFFFF, 0xF0F00000) == 0xF0F00000 ? 1 : -1];

//Intel
char _andn_u32_0[_andn_u32(0xFFFF1FFF, 0x0000F000) == 0x0000E000 ? 1 : -1];
char _andn_u32_1[_andn_u32(0xF0F0F0F0, 0x01010101) == 0x01010101 ? 1 : -1];

#ifdef __X86_64__
char andn_u64_0[__andn_u64(0x0000FFFF11110000, 0x0101010101010101) == 0x0101000000000101 ? 1 : -1];
char andn_u64_1[__andn_u64(0x0FFFFFFFFFFF137F, 0xF00000000000FFFF) == 0xF00000000000EC80 ? 1 : -1];

//Intel
char _andn_u64_0[_andn_u64(0xFFFF0000FFFF0000, 0x0000FFFF0000FFFF) == 0x0000FFFF0000FFFF ? 1 : -1];
char _andn_u64_1[_andn_u64(0xFFFFEDCBFFFFA987, 0x0000FFFF0000FF00) == 0x0000123400005600 ? 1 : -1];
#endif

//BEXTR
char bextr_u32_0[__bextr_u32(0xFFFF0000, 0x00001010) == 0x0000FFFF ? 1 : -1];
char bextr_u32_1[__bextr_u32(0x00FFF800, 0x0000100B) == 0x00001FFF ? 1 : -1];

//Intel
char _bextr_u32_0[_bextr_u32(0x10FFF800, 20, 9) == 0x0000010F ? 1 : -1];
char _bextr_u32_1[_bextr_u32(0x0000FF10, 16, 16) == 0x00000000 ? 1 : -1];

#ifdef __X86_64__
char bextr_u64_0[__bextr_u64(0x7FFF00001111FFFF, 0x00002020) == 0x000000007FFF0000 ? 1 : -1];
char bextr_u64_1[__bextr_u64(0xF0FFF800FFFF1111, 0x00004040) == 0x0000000000000000 ? 1 : -1];

//Intel
char _bextr_u64_0[_bextr_u64(0x7FFFFFFF10FF1111, 32, 32) == 0x000000007FFFFFFF ? 1 : -1];
char _bextr_u64_1[_bextr_u64(0x1111FFFF0000FF10, 48, 16) == 0x0000000000001111 ? 1 : -1];
#endif

//BLSI
char blsi_u32_0[__blsi_u32(0x0000FFF8) == 0x00000008 ? 1 : -1];
char blsi_u32_1[__blsi_u32(0x00FF0000) == 0x00010000 ? 1 : -1];

//Intel
char _blsi_u32_0[_blsi_u32(0x70000B00) == 0x00000100 ? 1 : -1];
char _blsi_u32_1[_blsi_u32(0x80000000) == 0x80000000 ? 1 : -1];

#ifdef __X86_64__
char blsi_u64_0[__blsi_u64(0xF0FFF800FFF00000) == 0x100000 ? 1 : -1];
char blsi_u64_1[__blsi_u64(0x0AE0000000000000) == 0x0020000000000000 ? 1 : -1];

//Intel
char _blsi_u64_0[_blsi_u64(0xFFFFFC0000000000) == 0x0000040000000000 ? 1 : -1];
char _blsi_u64_1[_blsi_u64(0x0FE0000000000000) == 0x0020000000000000 ? 1 : -1];
#endif

//BLSMSK
char blsmsk_u32_0[__blsmsk_u32(0xF0F0F0F0) == 0x0000001F ? 1 : -1];
char blsmsk_u32_1[__blsmsk_u32(0x7FFFFC00) == 0x000007FF ? 1 : -1];
 
//Intel
char _blsmsk_u32_0[_blsmsk_u32(0xB0000000) == 0x1FFFFFFF ? 1 : -1];
char _blsmsk_u32_1[_blsmsk_u32(0x00000000) == 0xFFFFFFFF ? 1 : -1];

#ifdef __X86_64__
char blsmsk_u64_0[__blsmsk_u64(0xF0F0F0F000A00000) == 0x00000000003FFFFF ? 1 : -1];
char blsmsk_u64_1[__blsmsk_u64(0x1111100000000800) == 0x0000000000000FFF ? 1 : -1];

//Intel
char _blsmsk_u64_0[_blsmsk_u64(0xFFFFFC0000000000) == 0x000007FFFFFFFFFF ? 1 : -1];
char _blsmsk_u64_1[_blsmsk_u64(0xFFFFFFFFFFFFFFFF) == 0x0000000000000001 ? 1 : -1];
#endif

//BLSR
char blsr_u32_0[__blsr_u32(0xB00FFFFF) == 0xB00FFFFE ? 1 : -1];
char blsr_u32_1[__blsr_u32(0xFFFFFFFF) == 0xFFFFFFFE ? 1 : -1];

//Intel
char _blsr_u32_0[_blsr_u32(0xB0000000) == 0xA0000000 ? 1 : -1];
char _blsr_u32_1[_blsr_u32(0x00000000) == 0x00000000 ? 1 : -1];

#ifdef __X86_64__
char blsr_u64_0[__blsr_u64(0xB00FF70000000000) == 0xB00FF60000000000 ? 1 : -1];
char blsr_u64_1[__blsr_u64(0xFFFFFFFFFFFFFFFF) == 0xFFFFFFFFFFFFFFFE ? 1 : -1];

//Intel
char _blsr_u64_0[_blsr_u64(0xB00FFFFFFFFF0000) == 0xB00FFFFFFFFE0000 ? 1 : -1];
char _blsr_u64_1[_blsr_u64(0x8000000000000000) == 0x0000000000000000 ? 1 : -1];
#endif //ifdef __X86_64__

#endif //!(defined(TEST_TZCNT)
#endif // __cplusplus
