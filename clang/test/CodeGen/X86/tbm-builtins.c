// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-unknown-unknown -target-feature +tbm -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -x c++ -std=c++11 -ffreestanding %s -triple=x86_64-unknown-unknown -target-feature +tbm -emit-llvm -o - | FileCheck %s

#include <x86intrin.h>

// NOTE: This should match the tests in llvm/test/CodeGen/X86/tbm-intrinsics-fast-isel.ll

unsigned int test__bextri_u32(unsigned int a) {
  // CHECK-LABEL: test__bextri_u32
  // CHECK: call i32 @llvm.x86.tbm.bextri.u32(i32 %{{.*}}, i32 1)
  return __bextri_u32(a, 1);
}

#ifdef __x86_64__
unsigned long long test__bextri_u64(unsigned long long a) {
  // CHECK-LABEL: test__bextri_u64
  // CHECK: call i64 @llvm.x86.tbm.bextri.u64(i64 %{{.*}}, i64 2)
  return __bextri_u64(a, 2);
}

unsigned long long test__bextri_u64_bigint(unsigned long long a) {
  // CHECK-LABEL: test__bextri_u64_bigint
  // CHECK: call i64 @llvm.x86.tbm.bextri.u64(i64 %{{.*}}, i64 549755813887)
  return __bextri_u64(a, 0x7fffffffffLL);
}
#endif

unsigned int test__blcfill_u32(unsigned int a) {
  // CHECK-LABEL: test__blcfill_u32
  // CHECK: [[TMP:%.*]] = add i32 %{{.*}}, 1
  // CHECK: %{{.*}} = and i32 %{{.*}}, [[TMP]]
  return __blcfill_u32(a);
}

#ifdef __x86_64__
unsigned long long test__blcfill_u64(unsigned long long a) {
  // CHECK-LABEL: test__blcfill_u64
  // CHECK: [[TMP:%.*]] = add i64 %{{.*}}, 1
  // CHECK: %{{.*}} = and i64 %{{.*}}, [[TMP]]
  return __blcfill_u64(a);
}
#endif

unsigned int test__blci_u32(unsigned int a) {
  // CHECK-LABEL: test__blci_u32
  // CHECK: [[TMP1:%.*]] = add i32 %{{.*}}, 1
  // CHECK: [[TMP2:%.*]] = xor i32 [[TMP1]], -1
  // CHECK: %{{.*}} = or i32 %{{.*}}, [[TMP2]]
  return __blci_u32(a);
}

#ifdef __x86_64__
unsigned long long test__blci_u64(unsigned long long a) {
  // CHECK-LABEL: test__blci_u64
  // CHECK: [[TMP1:%.*]] = add i64 %{{.*}}, 1
  // CHECK: [[TMP2:%.*]] = xor i64 [[TMP1]], -1
  // CHECK: %{{.*}} = or i64 %{{.*}}, [[TMP2]]
  return __blci_u64(a);
}
#endif

unsigned int test__blcic_u32(unsigned int a) {
  // CHECK-LABEL: test__blcic_u32
  // CHECK: [[TMP1:%.*]] = xor i32 %{{.*}}, -1
  // CHECK: [[TMP2:%.*]] = add i32 %{{.*}}, 1
  // CHECK-NEXT: {{.*}} = and i32 [[TMP1]], [[TMP2]]
  return __blcic_u32(a);
}

#ifdef __x86_64__
unsigned long long test__blcic_u64(unsigned long long a) {
  // CHECK-LABEL: test__blcic_u64
  // CHECK: [[TMP1:%.*]] = xor i64 %{{.*}}, -1
  // CHECK: [[TMP2:%.*]] = add i64 %{{.*}}, 1
  // CHECK-NEXT: {{.*}} = and i64 [[TMP1]], [[TMP2]]
  return __blcic_u64(a);
}
#endif

unsigned int test__blcmsk_u32(unsigned int a) {
  // CHECK-LABEL: test__blcmsk_u32
  // CHECK: [[TMP:%.*]] = add i32 %{{.*}}, 1
  // CHECK-NEXT: {{.*}} = xor i32 %{{.*}}, [[TMP]]
  return __blcmsk_u32(a);
}

#ifdef __x86_64__
unsigned long long test__blcmsk_u64(unsigned long long a) {
  // CHECK-LABEL: test__blcmsk_u64
  // CHECK: [[TMP:%.*]] = add i64 %{{.*}}, 1
  // CHECK-NEXT: {{.*}} = xor i64 %{{.*}}, [[TMP]]
  return __blcmsk_u64(a);
}
#endif

unsigned int test__blcs_u32(unsigned int a) {
  // CHECK-LABEL: test__blcs_u32
  // CHECK: [[TMP:%.*]] = add i32 %{{.*}}, 1
  // CHECK-NEXT: {{.*}} = or i32 %{{.*}}, [[TMP]]
  return __blcs_u32(a);
}

#ifdef __x86_64__
unsigned long long test__blcs_u64(unsigned long long a) {
  // CHECK-LABEL: test__blcs_u64
  // CHECK: [[TMP:%.*]] = add i64 %{{.*}}, 1
  // CHECK-NEXT: {{.*}} = or i64 %{{.*}}, [[TMP]]
  return __blcs_u64(a);
}
#endif

unsigned int test__blsfill_u32(unsigned int a) {
  // CHECK-LABEL: test__blsfill_u32
  // CHECK: [[TMP:%.*]] = sub i32 %{{.*}}, 1
  // CHECK-NEXT: {{.*}} = or i32 %{{.*}}, [[TMP]]
  return __blsfill_u32(a);
}

#ifdef __x86_64__
unsigned long long test__blsfill_u64(unsigned long long a) {
  // CHECK-LABEL: test__blsfill_u64
  // CHECK: [[TMP:%.*]] = sub i64 %{{.*}}, 1
  // CHECK-NEXT: {{.*}} = or i64 %{{.*}}, [[TMP]]
  return __blsfill_u64(a);
}
#endif

unsigned int test__blsic_u32(unsigned int a) {
  // CHECK-LABEL: test__blsic_u32
  // CHECK: [[TMP1:%.*]] = xor i32 %{{.*}}, -1
  // CHECK: [[TMP2:%.*]] = sub i32 %{{.*}}, 1
  // CHECK-NEXT: {{.*}} = or i32 [[TMP1]], [[TMP2]]
  return __blsic_u32(a);
}

#ifdef __x86_64__
unsigned long long test__blsic_u64(unsigned long long a) {
  // CHECK-LABEL: test__blsic_u64
  // CHECK: [[TMP1:%.*]] = xor i64 %{{.*}}, -1
  // CHECK: [[TMP2:%.*]] = sub i64 %{{.*}}, 1
  // CHECK-NEXT: {{.*}} = or i64 [[TMP1]], [[TMP2]]
  return __blsic_u64(a);
}
#endif

unsigned int test__t1mskc_u32(unsigned int a) {
  // CHECK-LABEL: test__t1mskc_u32
  // CHECK: [[TMP1:%.*]] = xor i32 %{{.*}}, -1
  // CHECK: [[TMP2:%.*]] = add i32 %{{.*}}, 1
  // CHECK-NEXT: {{.*}} = or i32 [[TMP1]], [[TMP2]]
  return __t1mskc_u32(a);
}

#ifdef __x86_64__
unsigned long long test__t1mskc_u64(unsigned long long a) {
  // CHECK-LABEL: test__t1mskc_u64
  // CHECK: [[TMP1:%.*]] = xor i64 %{{.*}}, -1
  // CHECK: [[TMP2:%.*]] = add i64 %{{.*}}, 1
  // CHECK-NEXT: {{.*}} = or i64 [[TMP1]], [[TMP2]]
  return __t1mskc_u64(a);
}
#endif

unsigned int test__tzmsk_u32(unsigned int a) {
  // CHECK-LABEL: test__tzmsk_u32
  // CHECK: [[TMP1:%.*]] = xor i32 %{{.*}}, -1
  // CHECK: [[TMP2:%.*]] = sub i32 %{{.*}}, 1
  // CHECK-NEXT: {{.*}} = and i32 [[TMP1]], [[TMP2]]
  return __tzmsk_u32(a);
}

#ifdef __x86_64__
unsigned long long test__tzmsk_u64(unsigned long long a) {
  // CHECK-LABEL: test__tzmsk_u64
  // CHECK: [[TMP1:%.*]] = xor i64 %{{.*}}, -1
  // CHECK: [[TMP2:%.*]] = sub i64 %{{.*}}, 1
  // CHECK-NEXT: {{.*}} = and i64 [[TMP1]], [[TMP2]]
  return __tzmsk_u64(a);
}
#endif

// Test constexpr handling.
#if defined(__cplusplus) && (__cplusplus >= 201103L)
char bextri32_0[__bextri_u32(0x00000000, 0x00000000) == 0x00000000 ? 1 : -1];
char bextri32_1[__bextri_u32(0x000003F0, 0xFFFF1004) == 0x0000003F ? 1 : -1];
char bextri32_2[__bextri_u32(0x000003F0, 0xFFFF3008) == 0x00000003 ? 1 : -1];

char blcfill32[__blcfill_u32(0x89ABCDEF) == (0x89ABCDEF & (0x89ABCDEF + 1)) ? 1 : -1];
char blci32[__blci_u32(0x89ABCDEF) == (0x89ABCDEF | ~(0x89ABCDEF + 1)) ? 1 : -1];
char blcmsk32[__blcmsk_u32(0x89ABCDEF) == (0x89ABCDEF ^ (0x89ABCDEF + 1)) ? 1 : -1];
char blcs32[__blcs_u32(0x89ABCDEF) == (0x89ABCDEF | (0x89ABCDEF + 1)) ? 1 : -1];
char blsfill32[__blsfill_u32(0x89ABCDEF) == (0x89ABCDEF | (0x89ABCDEF - 1)) ? 1 : -1];
char blsic32[__blsic_u32(0x89ABCDEF) == (~0x89ABCDEF | (0x89ABCDEF - 1)) ? 1 : -1];
char t1mskc32[__t1mskc_u32(0x89ABCDEF) == (~0x89ABCDEF | (0x89ABCDEF + 1)) ? 1 : -1];
char tzmsk32[__tzmsk_u32(0x89ABCDEF) == (~0x89ABCDEF & (0x89ABCDEF - 1)) ? 1 : -1];

#ifdef __x86_64__
char bextri64_0[__bextri_u64(0x0000000000000000ULL, 0x0000000000000000ULL) == 0x0000000000000000ULL ? 1 : -1];
char bextri64_1[__bextri_u64(0xF000000000000001ULL, 0x0000000000004001ULL) == 0x7800000000000000ULL ? 1 : -1];
char bextri64_2[__bextri_u64(0xF000000000000001ULL, 0xFFFFFFFFFFFF1001ULL) == 0x0000000000000000ULL ? 1 : -1];

char blcfill64[__blcfill_u64(0xFEDCBA9876543210) == (0xFEDCBA9876543210 & (0xFEDCBA9876543210 + 1)) ? 1 : -1];
char blci64[__blci_u64(0xFEDCBA9876543210) == (0xFEDCBA9876543210 | ~(0xFEDCBA9876543210 + 1)) ? 1 : -1];
char blcmsk64[__blcmsk_u64(0xFEDCBA9876543210) == (0xFEDCBA9876543210 ^ (0xFEDCBA9876543210 + 1)) ? 1 : -1];
char blcs64[__blcs_u64(0xFEDCBA9876543210) == (0xFEDCBA9876543210 | (0xFEDCBA9876543210 + 1)) ? 1 : -1];
char blsfill64[__blsfill_u64(0xFEDCBA9876543210) == (0xFEDCBA9876543210 | (0xFEDCBA9876543210 - 1)) ? 1 : -1];
char blsic64[__blsic_u64(0xFEDCBA9876543210) == (~0xFEDCBA9876543210 | (0xFEDCBA9876543210 - 1)) ? 1 : -1];
char t1mskc64[__t1mskc_u64(0xFEDCBA9876543210) == (~0xFEDCBA9876543210 | (0xFEDCBA9876543210 + 1)) ? 1 : -1];
char tzmsk64[__tzmsk_u64(0xFEDCBA9876543210) == (~0xFEDCBA9876543210 & (0xFEDCBA9876543210 - 1)) ? 1 : -1];
#endif
#endif
