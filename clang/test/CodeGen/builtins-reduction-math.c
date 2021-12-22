// RUN: %clang_cc1 -triple x86_64-apple-darwin %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s

typedef float float4 __attribute__((ext_vector_type(4)));
typedef short int si8 __attribute__((ext_vector_type(8)));
typedef unsigned int u4 __attribute__((ext_vector_type(4)));

__attribute__((address_space(1))) float4 vf1_as_one;

void test_builtin_reduce_max(float4 vf1, si8 vi1, u4 vu1) {
  // CHECK-LABEL: define void @test_builtin_reduce_max(
  // CHECK:      [[VF1:%.+]] = load <4 x float>, <4 x float>* %vf1.addr, align 16
  // CHECK-NEXT: call float @llvm.vector.reduce.fmax.v4f32(<4 x float> [[VF1]])
  float r1 = __builtin_reduce_max(vf1);

  // CHECK:      [[VI1:%.+]] = load <8 x i16>, <8 x i16>* %vi1.addr, align 16
  // CHECK-NEXT: call i16 @llvm.vector.reduce.smax.v8i16(<8 x i16> [[VI1]])
  short r2 = __builtin_reduce_max(vi1);

  // CHECK:      [[VU1:%.+]] = load <4 x i32>, <4 x i32>* %vu1.addr, align 16
  // CHECK-NEXT: call i32 @llvm.vector.reduce.umax.v4i32(<4 x i32> [[VU1]])
  unsigned r3 = __builtin_reduce_max(vu1);

  // CHECK:      [[VF1_AS1:%.+]] = load <4 x float>, <4 x float> addrspace(1)* @vf1_as_one, align 16
  // CHECK-NEXT: [[RDX1:%.+]] = call float @llvm.vector.reduce.fmax.v4f32(<4 x float> [[VF1_AS1]])
  // CHECK-NEXT: fpext float [[RDX1]] to double
  const double r4 = __builtin_reduce_max(vf1_as_one);

  // CHECK:      [[CVI1:%.+]] = load <8 x i16>, <8 x i16>* %cvi1, align 16
  // CHECK-NEXT: [[RDX2:%.+]] = call i16 @llvm.vector.reduce.smax.v8i16(<8 x i16> [[CVI1]])
  // CHECK-NEXT: sext i16 [[RDX2]] to i64
  const si8 cvi1 = vi1;
  unsigned long long r5 = __builtin_reduce_max(cvi1);
}

void test_builtin_reduce_min(float4 vf1, si8 vi1, u4 vu1) {
  // CHECK-LABEL: define void @test_builtin_reduce_min(
  // CHECK:      [[VF1:%.+]] = load <4 x float>, <4 x float>* %vf1.addr, align 16
  // CHECK-NEXT: call float @llvm.vector.reduce.fmin.v4f32(<4 x float> [[VF1]])
  float r1 = __builtin_reduce_min(vf1);

  // CHECK:      [[VI1:%.+]] = load <8 x i16>, <8 x i16>* %vi1.addr, align 16
  // CHECK-NEXT: call i16 @llvm.vector.reduce.smin.v8i16(<8 x i16> [[VI1]])
  short r2 = __builtin_reduce_min(vi1);

  // CHECK:      [[VU1:%.+]] = load <4 x i32>, <4 x i32>* %vu1.addr, align 16
  // CHECK-NEXT: call i32 @llvm.vector.reduce.umin.v4i32(<4 x i32> [[VU1]])
  unsigned r3 = __builtin_reduce_min(vu1);

  // CHECK:      [[VF1_AS1:%.+]] = load <4 x float>, <4 x float> addrspace(1)* @vf1_as_one, align 16
  // CHECK-NEXT: [[RDX1:%.+]] = call float @llvm.vector.reduce.fmin.v4f32(<4 x float> [[VF1_AS1]])
  // CHECK-NEXT: fpext float [[RDX1]] to double
  const double r4 = __builtin_reduce_min(vf1_as_one);

  // CHECK:      [[CVI1:%.+]] = load <8 x i16>, <8 x i16>* %cvi1, align 16
  // CHECK-NEXT: [[RDX2:%.+]] = call i16 @llvm.vector.reduce.smin.v8i16(<8 x i16> [[CVI1]])
  // CHECK-NEXT: sext i16 [[RDX2]] to i64
  const si8 cvi1 = vi1;
  unsigned long long r5 = __builtin_reduce_min(cvi1);
}

void test_builtin_reduce_xor(si8 vi1, u4 vu1) {

  // CHECK:      [[VI1:%.+]] = load <8 x i16>, <8 x i16>* %vi1.addr, align 16
  // CHECK-NEXT: call i16 @llvm.vector.reduce.xor.v8i16(<8 x i16> [[VI1]])
  short r2 = __builtin_reduce_xor(vi1);

  // CHECK:      [[VU1:%.+]] = load <4 x i32>, <4 x i32>* %vu1.addr, align 16
  // CHECK-NEXT: call i32 @llvm.vector.reduce.xor.v4i32(<4 x i32> [[VU1]])
  unsigned r3 = __builtin_reduce_xor(vu1);
}
