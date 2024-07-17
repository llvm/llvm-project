// RUN: %clang_cc1 -triple x86_64-apple-darwin %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -O1 -triple aarch64 -target-feature +sve  %s -emit-llvm -disable-llvm-passes -o - | FileCheck --check-prefixes=SVE   %s

typedef float float4 __attribute__((ext_vector_type(4)));
typedef short int si8 __attribute__((ext_vector_type(8)));
typedef unsigned int u4 __attribute__((ext_vector_type(4)));

__attribute__((address_space(1))) float4 vf1_as_one;

void test_builtin_reduce_max(float4 vf1, si8 vi1, u4 vu1) {
  // CHECK-LABEL: define void @test_builtin_reduce_max(
  // CHECK:      [[VF1:%.+]] = load <4 x float>, ptr %vf1.addr, align 16
  // CHECK-NEXT: call float @llvm.vector.reduce.fmax.v4f32(<4 x float> [[VF1]])
  float r1 = __builtin_reduce_max(vf1);

  // CHECK:      [[VI1:%.+]] = load <8 x i16>, ptr %vi1.addr, align 16
  // CHECK-NEXT: call i16 @llvm.vector.reduce.smax.v8i16(<8 x i16> [[VI1]])
  short r2 = __builtin_reduce_max(vi1);

  // CHECK:      [[VU1:%.+]] = load <4 x i32>, ptr %vu1.addr, align 16
  // CHECK-NEXT: call i32 @llvm.vector.reduce.umax.v4i32(<4 x i32> [[VU1]])
  unsigned r3 = __builtin_reduce_max(vu1);

  // CHECK:      [[VF1_AS1:%.+]] = load <4 x float>, ptr addrspace(1) @vf1_as_one, align 16
  // CHECK-NEXT: [[RDX1:%.+]] = call float @llvm.vector.reduce.fmax.v4f32(<4 x float> [[VF1_AS1]])
  // CHECK-NEXT: fpext float [[RDX1]] to double
  const double r4 = __builtin_reduce_max(vf1_as_one);

  // CHECK:      [[CVI1:%.+]] = load <8 x i16>, ptr %cvi1, align 16
  // CHECK-NEXT: [[RDX2:%.+]] = call i16 @llvm.vector.reduce.smax.v8i16(<8 x i16> [[CVI1]])
  // CHECK-NEXT: sext i16 [[RDX2]] to i64
  const si8 cvi1 = vi1;
  unsigned long long r5 = __builtin_reduce_max(cvi1);
}

void test_builtin_reduce_min(float4 vf1, si8 vi1, u4 vu1) {
  // CHECK-LABEL: define void @test_builtin_reduce_min(
  // CHECK:      [[VF1:%.+]] = load <4 x float>, ptr %vf1.addr, align 16
  // CHECK-NEXT: call float @llvm.vector.reduce.fmin.v4f32(<4 x float> [[VF1]])
  float r1 = __builtin_reduce_min(vf1);

  // CHECK:      [[VI1:%.+]] = load <8 x i16>, ptr %vi1.addr, align 16
  // CHECK-NEXT: call i16 @llvm.vector.reduce.smin.v8i16(<8 x i16> [[VI1]])
  short r2 = __builtin_reduce_min(vi1);

  // CHECK:      [[VU1:%.+]] = load <4 x i32>, ptr %vu1.addr, align 16
  // CHECK-NEXT: call i32 @llvm.vector.reduce.umin.v4i32(<4 x i32> [[VU1]])
  unsigned r3 = __builtin_reduce_min(vu1);

  // CHECK:      [[VF1_AS1:%.+]] = load <4 x float>, ptr addrspace(1) @vf1_as_one, align 16
  // CHECK-NEXT: [[RDX1:%.+]] = call float @llvm.vector.reduce.fmin.v4f32(<4 x float> [[VF1_AS1]])
  // CHECK-NEXT: fpext float [[RDX1]] to double
  const double r4 = __builtin_reduce_min(vf1_as_one);

  // CHECK:      [[CVI1:%.+]] = load <8 x i16>, ptr %cvi1, align 16
  // CHECK-NEXT: [[RDX2:%.+]] = call i16 @llvm.vector.reduce.smin.v8i16(<8 x i16> [[CVI1]])
  // CHECK-NEXT: sext i16 [[RDX2]] to i64
  const si8 cvi1 = vi1;
  unsigned long long r5 = __builtin_reduce_min(cvi1);
}

void test_builtin_reduce_add(si8 vi1, u4 vu1) {
  // CHECK:      [[VI1:%.+]] = load <8 x i16>, ptr %vi1.addr, align 16
  // CHECK-NEXT: call i16 @llvm.vector.reduce.add.v8i16(<8 x i16> [[VI1]])
  short r2 = __builtin_reduce_add(vi1);

  // CHECK:      [[VU1:%.+]] = load <4 x i32>, ptr %vu1.addr, align 16
  // CHECK-NEXT: call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> [[VU1]])
  unsigned r3 = __builtin_reduce_add(vu1);

  // CHECK:      [[CVI1:%.+]] = load <8 x i16>, ptr %cvi1, align 16
  // CHECK-NEXT: [[RDX1:%.+]] = call i16 @llvm.vector.reduce.add.v8i16(<8 x i16> [[CVI1]])
  // CHECK-NEXT: sext i16 [[RDX1]] to i32
  const si8 cvi1 = vi1;
  int r4 = __builtin_reduce_add(cvi1);

  // CHECK:      [[CVU1:%.+]] = load <4 x i32>, ptr %cvu1, align 16
  // CHECK-NEXT: [[RDX2:%.+]] = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> [[CVU1]])
  // CHECK-NEXT: zext i32 [[RDX2]] to i64
  const u4 cvu1 = vu1;
  unsigned long long r5 = __builtin_reduce_add(cvu1);
}

void test_builtin_reduce_mul(si8 vi1, u4 vu1) {
  // CHECK:      [[VI1:%.+]] = load <8 x i16>, ptr %vi1.addr, align 16
  // CHECK-NEXT: call i16 @llvm.vector.reduce.mul.v8i16(<8 x i16> [[VI1]])
  short r2 = __builtin_reduce_mul(vi1);

  // CHECK:      [[VU1:%.+]] = load <4 x i32>, ptr %vu1.addr, align 16
  // CHECK-NEXT: call i32 @llvm.vector.reduce.mul.v4i32(<4 x i32> [[VU1]])
  unsigned r3 = __builtin_reduce_mul(vu1);

  // CHECK:      [[CVI1:%.+]] = load <8 x i16>, ptr %cvi1, align 16
  // CHECK-NEXT: [[RDX1:%.+]] = call i16 @llvm.vector.reduce.mul.v8i16(<8 x i16> [[CVI1]])
  // CHECK-NEXT: sext i16 [[RDX1]] to i32
  const si8 cvi1 = vi1;
  int r4 = __builtin_reduce_mul(cvi1);

  // CHECK:      [[CVU1:%.+]] = load <4 x i32>, ptr %cvu1, align 16
  // CHECK-NEXT: [[RDX2:%.+]] = call i32 @llvm.vector.reduce.mul.v4i32(<4 x i32> [[CVU1]])
  // CHECK-NEXT: zext i32 [[RDX2]] to i64
  const u4 cvu1 = vu1;
  unsigned long long r5 = __builtin_reduce_mul(cvu1);
}

void test_builtin_reduce_xor(si8 vi1, u4 vu1) {

  // CHECK:      [[VI1:%.+]] = load <8 x i16>, ptr %vi1.addr, align 16
  // CHECK-NEXT: call i16 @llvm.vector.reduce.xor.v8i16(<8 x i16> [[VI1]])
  short r2 = __builtin_reduce_xor(vi1);

  // CHECK:      [[VU1:%.+]] = load <4 x i32>, ptr %vu1.addr, align 16
  // CHECK-NEXT: call i32 @llvm.vector.reduce.xor.v4i32(<4 x i32> [[VU1]])
  unsigned r3 = __builtin_reduce_xor(vu1);
}

void test_builtin_reduce_or(si8 vi1, u4 vu1) {

  // CHECK:      [[VI1:%.+]] = load <8 x i16>, ptr %vi1.addr, align 16
  // CHECK-NEXT: call i16 @llvm.vector.reduce.or.v8i16(<8 x i16> [[VI1]])
  short r2 = __builtin_reduce_or(vi1);

  // CHECK:      [[VU1:%.+]] = load <4 x i32>, ptr %vu1.addr, align 16
  // CHECK-NEXT: call i32 @llvm.vector.reduce.or.v4i32(<4 x i32> [[VU1]])
  unsigned r3 = __builtin_reduce_or(vu1);
}

void test_builtin_reduce_and(si8 vi1, u4 vu1) {

  // CHECK:      [[VI1:%.+]] = load <8 x i16>, ptr %vi1.addr, align 16
  // CHECK-NEXT: call i16 @llvm.vector.reduce.and.v8i16(<8 x i16> [[VI1]])
  short r2 = __builtin_reduce_and(vi1);

  // CHECK:      [[VU1:%.+]] = load <4 x i32>, ptr %vu1.addr, align 16
  // CHECK-NEXT: call i32 @llvm.vector.reduce.and.v4i32(<4 x i32> [[VU1]])
  unsigned r3 = __builtin_reduce_and(vu1);
}

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>

void test_builtin_reduce_SVE(int a, unsigned long long b, short c, float d) {
  // SVE-LABEL: void @test_builtin_reduce_SVE(

  svint32_t vec_a = svdup_s32(a);
  svuint64_t vec_b = svdup_u64(b);
  svint16_t vec_c1 = svdup_s16(c);
  svuint16_t vec_c2 = svdup_u16(c);
  svfloat32_t vec_d = svdup_f32(d);

  // SVE:      [[VF1:%.+]] = load <vscale x 4 x i32>, ptr %vec_a
  // SVE-NEXT: call i32 @llvm.vector.reduce.add.nxv4i32(<vscale x 4 x i32> [[VF1]])
  int r1 = __builtin_reduce_add(vec_a);

  // SVE:      [[VF2:%.+]] = load <vscale x 4 x i32>, ptr %vec_a
  // SVE-NEXT: call i32 @llvm.vector.reduce.mul.nxv4i32(<vscale x 4 x i32> [[VF2]])
  int r2 = __builtin_reduce_mul(vec_a);

  // SVE:      [[VF3:%.+]] = load <vscale x 2 x i64>, ptr %vec_b
  // SVE-NEXT: call i64 @llvm.vector.reduce.xor.nxv2i64(<vscale x 2 x i64> [[VF3]])
  long long r3 = __builtin_reduce_xor(vec_b);

  // SVE:      [[VF4:%.+]] = load <vscale x 2 x i64>, ptr %vec_b
  // SVE-NEXT: call i64 @llvm.vector.reduce.or.nxv2i64(<vscale x 2 x i64> [[VF4]])
  long long r4 = __builtin_reduce_or(vec_b);

  // SVE:      [[VF5:%.+]] = load <vscale x 2 x i64>, ptr %vec_b
  // SVE-NEXT: call i64 @llvm.vector.reduce.and.nxv2i64(<vscale x 2 x i64> [[VF5]])
  long long r5 = __builtin_reduce_and(vec_b);

  // SVE:      [[VF6:%.+]] = load <vscale x 8 x i16>, ptr %vec_c1
  // SVE-NEXT: call i16 @llvm.vector.reduce.smax.nxv8i16(<vscale x 8 x i16> [[VF6]])
  short r6 = __builtin_reduce_max(vec_c1);

  // SVE:      [[VF7:%.+]] = load <vscale x 8 x i16>, ptr %vec_c2
  // SVE-NEXT: call i16 @llvm.vector.reduce.umin.nxv8i16(<vscale x 8 x i16> [[VF7]])
  unsigned short r7 = __builtin_reduce_min(vec_c2);

  // SVE:      [[VF8:%.+]] = load <vscale x 4 x float>, ptr %vec_d
  // SVE-NEXT: call float @llvm.vector.reduce.fmax.nxv4f32(<vscale x 4 x float> [[VF8]])
  float r8 = __builtin_reduce_max(vec_d);

  // SVE:      [[VF9:%.+]] = load <vscale x 4 x float>, ptr %vec_d
  // SVE-NEXT: call float @llvm.vector.reduce.fmin.nxv4f32(<vscale x 4 x float> [[VF9]])
  float r9 = __builtin_reduce_min(vec_d);
}
#endif
