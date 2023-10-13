// RUN: %clang_cc1 -O1 -triple aarch64 -target-feature +neon %s -emit-llvm -o - | FileCheck --check-prefixes=CHECK,NEON %s
// RUN: %clang_cc1 -O1 -triple aarch64 -target-feature +sve  %s -emit-llvm -o - | FileCheck --check-prefixes=CHECK,SVE  %s
// RUN: %clang_cc1 -O1 -triple riscv64 -target-feature +v    %s -emit-llvm -o - | FileCheck --check-prefixes=CHECK,RISCV  %s

// Note that this does not make sense to check for x86 SIMD types, because
// __m128i, __m256i, and __m512i do not specify the element type. There are no
// "logical" number of elements in them.

typedef int int1 __attribute__((vector_size(4)));
typedef int int4 __attribute__((vector_size(16)));
typedef int int8 __attribute__((vector_size(32)));
typedef int int16 __attribute__((vector_size(64)));
typedef float float2 __attribute__((vector_size(8)));
typedef long extLong4 __attribute__((ext_vector_type(4)));


int test_builtin_vectorelements_int1() {
  // CHECK-LABEL: i32 @test_builtin_vectorelements_int1(
  // CHECK: ret i32 1
  return __builtin_vectorelements(int1);
}

int test_builtin_vectorelements_int4() {
  // CHECK-LABEL: i32 @test_builtin_vectorelements_int4(
  // CHECK: ret i32 4
  return __builtin_vectorelements(int4);
}

int test_builtin_vectorelements_int8() {
  // CHECK-LABEL: i32 @test_builtin_vectorelements_int8(
  // CHECK: ret i32 8
  return __builtin_vectorelements(int8);
}

int test_builtin_vectorelements_int16() {
  // CHECK-LABEL: i32 @test_builtin_vectorelements_int16(
  // CHECK: ret i32 16
  return __builtin_vectorelements(int16);
}

int test_builtin_vectorelements_float2() {
  // CHECK-LABEL: i32 @test_builtin_vectorelements_float2(
  // CHECK: ret i32 2
  return __builtin_vectorelements(float2);
}

int test_builtin_vectorelements_extLong4() {
  // CHECK-LABEL: i32 @test_builtin_vectorelements_extLong4(
  // CHECK: ret i32 4
  return __builtin_vectorelements(extLong4);
}

int test_builtin_vectorelements_multiply_constant() {
  // CHECK-LABEL: i32 @test_builtin_vectorelements_multiply_constant(
  // CHECK: ret i32 32
  return __builtin_vectorelements(int16) * 2;
}


#if defined(__ARM_NEON)
#include <arm_neon.h>

int test_builtin_vectorelements_neon32x4() {
  // NEON: i32 @test_builtin_vectorelements_neon32x4(
  // NEON: ret i32 4
  return __builtin_vectorelements(uint32x4_t);
}

int test_builtin_vectorelements_neon64x1() {
  // NEON: i32 @test_builtin_vectorelements_neon64x1(
  // NEON: ret i32 1
  return __builtin_vectorelements(uint64x1_t);
}
#endif

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>

int test_builtin_vectorelements_sve32() {
  // SVE: i32 @test_builtin_vectorelements_sve32(
  // SVE: [[VSCALE:%.+]] = tail call i32 @llvm.vscale.i32()
  // SVE: [[RES:%.+]] = shl nuw nsw i32 [[VSCALE]], 2
  // SVE: ret i32 [[RES]]
  return __builtin_vectorelements(svuint32_t);
}

int test_builtin_vectorelements_sve16() {
  // SVE: i32 @test_builtin_vectorelements_sve16(
  // SVE: [[VSCALE:%.+]] = tail call i32 @llvm.vscale.i32()
  // SVE: [[RES:%.+]] = shl nuw nsw i32 [[VSCALE]], 4
  // SVE: ret i32 [[RES]]
  return __builtin_vectorelements(svuint8_t);
}
#endif

#if defined(__riscv)
#include <riscv_vector.h>

int test_builtin_vectorelements_riscv8() {
  // RISCV: i32 @test_builtin_vectorelements_riscv8(
  // RISCV: [[VSCALE:%.+]] = tail call i32 @llvm.vscale.i32()
  // RISCV: [[RES:%.+]] = shl nuw nsw i32 [[VSCALE]], 3
  // RISCV: ret i32 [[RES]]
  return __builtin_vectorelements(vuint8m1_t);
}

int test_builtin_vectorelements_riscv64() {
  // RISCV: i32 @test_builtin_vectorelements_riscv64(
  // RISCV: [[VSCALE:%.+]] = tail call i32 @llvm.vscale.i32()
  // RISCV: ret i32 [[VSCALE]]
  return __builtin_vectorelements(vuint64m1_t);
}

int test_builtin_vectorelements_riscv32m2() {
  // RISCV: i32 @test_builtin_vectorelements_riscv32m2(
  // RISCV: [[VSCALE:%.+]] = tail call i32 @llvm.vscale.i32()
  // RISCV: [[RES:%.+]] = shl nuw nsw i32 [[VSCALE]], 2
  // RISCV: ret i32 [[RES]]
  return __builtin_vectorelements(vuint32m2_t);
}
#endif
