// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c2y -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c2y -isystem %S/Inputs -DTEST_C2Y_LIB_SPELLINGS %s -emit-llvm -o - | FileCheck %s --check-prefix=C2Y

#ifndef TEST_C2Y_LIB_SPELLINGS

// N=0 and N=1: no-op, no library call or bswap emitted.
// CHECK-LABEL: test_memreverse8_const0
// CHECK-NOT: call void @stdc_memreverse8
void test_memreverse8_const0(unsigned char *p) {
  __builtin_stdc_memreverse8(0, p);
}

// CHECK-LABEL: test_memreverse8_const1
// CHECK-NOT: call void @stdc_memreverse8
void test_memreverse8_const1(unsigned char *p) {
  __builtin_stdc_memreverse8(1, p);
}

// N=0 and N=1 with side effects: pointer argument is evaluated but no reversal.
// CHECK-LABEL: test_memreverse8_sideeffect_n0
// CHECK: getelementptr
// CHECK: store ptr
// CHECK-NOT: call void @stdc_memreverse8
unsigned char *gp;
void test_memreverse8_sideeffect_n0(void) {
  __builtin_stdc_memreverse8(0, gp++);
}

// CHECK-LABEL: test_memreverse8_sideeffect_n1
// CHECK: getelementptr
// CHECK: store ptr
// CHECK-NOT: call void @stdc_memreverse8
void test_memreverse8_sideeffect_n1(void) {
  __builtin_stdc_memreverse8(1, gp++);
}

// N=2, 4, 8: lowered to a single bswap with unaligned load/store.
// CHECK-LABEL: test_memreverse8_const2
// CHECK: load i16, ptr {{.*}}, align 1
// CHECK: call i16 @llvm.bswap.i16
// CHECK: store i16 {{.*}}, ptr {{.*}}, align 1
void test_memreverse8_const2(unsigned char *p) {
  __builtin_stdc_memreverse8(2, p);
}

// CHECK-LABEL: test_memreverse8_const4
// CHECK: load i32, ptr {{.*}}, align 1
// CHECK: call i32 @llvm.bswap.i32
// CHECK: store i32 {{.*}}, ptr {{.*}}, align 1
void test_memreverse8_const4(unsigned char *p) {
  __builtin_stdc_memreverse8(4, p);
}

// CHECK-LABEL: test_memreverse8_const8
// CHECK: load i64, ptr {{.*}}, align 1
// CHECK: call i64 @llvm.bswap.i64
// CHECK: store i64 {{.*}}, ptr {{.*}}, align 1
void test_memreverse8_const8(unsigned char *p) {
  __builtin_stdc_memreverse8(8, p);
}

// Constant N=3: not bswap-optimized, falls back to library call.
// CHECK-LABEL: test_memreverse8_const3
// CHECK: call void @stdc_memreverse8(
// CHECK-NOT: @llvm.bswap
void test_memreverse8_const3(unsigned char *p) {
  __builtin_stdc_memreverse8(3, p);
}

// Runtime N: falls back to library call.
// CHECK-LABEL: test_memreverse8_runtime
// CHECK: call void @stdc_memreverse8(
void test_memreverse8_runtime(__SIZE_TYPE__ n, unsigned char *p) {
  __builtin_stdc_memreverse8(n, p);
}

#endif // !TEST_C2Y_LIB_SPELLINGS

#ifdef TEST_C2Y_LIB_SPELLINGS
#include <stdbit.h>

// u8 is a no-op: single byte, nothing to swap.
// C2Y-LABEL: test_typed_memreverse8u8
// C2Y-NOT:   @llvm.bswap
unsigned char test_typed_memreverse8u8(unsigned char x) {
  return stdc_memreverse8u8(x);
}

// C2Y-LABEL: test_typed_memreverse8u16
// C2Y: call i16 @llvm.bswap.i16
unsigned short test_typed_memreverse8u16(unsigned short x) {
  return stdc_memreverse8u16(x);
}

// C2Y-LABEL: test_typed_memreverse8u32
// C2Y: call i32 @llvm.bswap.i32
unsigned int test_typed_memreverse8u32(unsigned int x) {
  return stdc_memreverse8u32(x);
}

// C2Y-LABEL: test_typed_memreverse8u64
// C2Y: call i64 @llvm.bswap.i64
__UINT64_TYPE__ test_typed_memreverse8u64(__UINT64_TYPE__ x) {
  return stdc_memreverse8u64(x);
}

#endif // TEST_C2Y_LIB_SPELLINGS
