// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s

typedef float float4 __attribute__((ext_vector_type(4)));
typedef short int si8 __attribute__((ext_vector_type(8)));
typedef unsigned int u4 __attribute__((ext_vector_type(4)));

__attribute__((address_space(1))) int int_as_one;
typedef int bar;
bar b;

void test_builtin_elementwise_abs(float f1, float f2, double d1, double d2,
                                  float4 vf1, float4 vf2, si8 vi1, si8 vi2,
                                  long long int i1, long long int i2, short si,
                                  _BitInt(31) bi1, _BitInt(31) bi2) {
  // CHECK-LABEL: define void @test_builtin_elementwise_abs(
  // CHECK:      [[F1:%.+]] = load float, float* %f1.addr, align 4
  // CHECK-NEXT:  call float @llvm.fabs.f32(float [[F1]])
  f2 = __builtin_elementwise_abs(f1);

  // CHECK:      [[D1:%.+]] = load double, double* %d1.addr, align 8
  // CHECK-NEXT: call double @llvm.fabs.f64(double [[D1]])
  d2 = __builtin_elementwise_abs(d1);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, <4 x float>* %vf1.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.fabs.v4f32(<4 x float> [[VF1]])
  vf2 = __builtin_elementwise_abs(vf1);

  // CHECK:      [[I1:%.+]] = load i64, i64* %i1.addr, align 8
  // CHECK-NEXT: call i64 @llvm.abs.i64(i64 [[I1]], i1 false)
  i2 = __builtin_elementwise_abs(i1);

  // CHECK:      [[VI1:%.+]] = load <8 x i16>, <8 x i16>* %vi1.addr, align 16
  // CHECK-NEXT: call <8 x i16> @llvm.abs.v8i16(<8 x i16> [[VI1]], i1 false)
  vi2 = __builtin_elementwise_abs(vi1);

  // CHECK:      [[CVI2:%.+]] = load <8 x i16>, <8 x i16>* %cvi2, align 16
  // CHECK-NEXT: call <8 x i16> @llvm.abs.v8i16(<8 x i16> [[CVI2]], i1 false)
  const si8 cvi2 = vi2;
  vi2 = __builtin_elementwise_abs(cvi2);

  // CHECK:      [[BI1:%.+]] = load i31, i31* %bi1.addr, align 4
  // CHECK-NEXT: call i31 @llvm.abs.i31(i31 [[BI1]], i1 false)
  bi2 = __builtin_elementwise_abs(bi1);

  // CHECK:      [[IA1:%.+]] = load i32, i32 addrspace(1)* @int_as_one, align 4
  // CHECK-NEXT: call i32 @llvm.abs.i32(i32 [[IA1]], i1 false)
  b = __builtin_elementwise_abs(int_as_one);

  // CHECK:   call i32 @llvm.abs.i32(i32 -10, i1 false)
  b = __builtin_elementwise_abs(-10);

  // CHECK:      [[SI:%.+]] = load i16, i16* %si.addr, align 2
  // CHECK-NEXT: [[SI_EXT:%.+]] = sext i16 [[SI]] to i32
  // CHECK-NEXT: [[RES:%.+]] = call i32 @llvm.abs.i32(i32 [[SI_EXT]], i1 false)
  // CHECK-NEXT: = trunc i32 [[RES]] to i16
  si = __builtin_elementwise_abs(si);
}

void test_builtin_elementwise_add_sat(float f1, float f2, double d1, double d2,
                                      float4 vf1, float4 vf2, long long int i1,
                                      long long int i2, si8 vi1, si8 vi2,
                                      unsigned u1, unsigned u2, u4 vu1, u4 vu2,
                                      _BitInt(31) bi1, _BitInt(31) bi2,
                                      unsigned _BitInt(55) bu1, unsigned _BitInt(55) bu2) {
  // CHECK:      [[I1:%.+]] = load i64, i64* %i1.addr, align 8
  // CHECK-NEXT: [[I2:%.+]] = load i64, i64* %i2.addr, align 8
  // CHECK-NEXT: call i64 @llvm.sadd.sat.i64(i64 [[I1]], i64 [[I2]])
  i1 = __builtin_elementwise_add_sat(i1, i2);

  // CHECK:      [[I1:%.+]] = load i64, i64* %i1.addr, align 8
  // CHECK-NEXT: call i64 @llvm.sadd.sat.i64(i64 [[I1]], i64 10)
  i1 = __builtin_elementwise_add_sat(i1, 10);

  // CHECK:      [[VI1:%.+]] = load <8 x i16>, <8 x i16>* %vi1.addr, align 16
  // CHECK-NEXT: [[VI2:%.+]] = load <8 x i16>, <8 x i16>* %vi2.addr, align 16
  // CHECK-NEXT: call <8 x i16> @llvm.sadd.sat.v8i16(<8 x i16> [[VI1]], <8 x i16> [[VI2]])
  vi1 = __builtin_elementwise_add_sat(vi1, vi2);

  // CHECK:      [[U1:%.+]] = load i32, i32* %u1.addr, align 4
  // CHECK-NEXT: [[U2:%.+]] = load i32, i32* %u2.addr, align 4
  // CHECK-NEXT: call i32 @llvm.uadd.sat.i32(i32 [[U1]], i32 [[U2]])
  u1 = __builtin_elementwise_add_sat(u1, u2);

  // CHECK:      [[VU1:%.+]] = load <4 x i32>, <4 x i32>* %vu1.addr, align 16
  // CHECK-NEXT: [[VU2:%.+]] = load <4 x i32>, <4 x i32>* %vu2.addr, align 16
  // CHECK-NEXT: call <4 x i32> @llvm.uadd.sat.v4i32(<4 x i32> [[VU1]], <4 x i32> [[VU2]])
  vu1 = __builtin_elementwise_add_sat(vu1, vu2);

  // CHECK:      [[BI1:%.+]] = load i31, i31* %bi1.addr, align 4
  // CHECK-NEXT: [[BI2:%.+]] = load i31, i31* %bi2.addr, align 4
  // CHECK-NEXT: call i31 @llvm.sadd.sat.i31(i31 [[BI1]], i31 [[BI2]])
  bi1 = __builtin_elementwise_add_sat(bi1, bi2);

  // CHECK:      [[BU1:%.+]] = load i55, i55* %bu1.addr, align 8
  // CHECK-NEXT: [[BU2:%.+]] = load i55, i55* %bu2.addr, align 8
  // CHECK-NEXT: call i55 @llvm.uadd.sat.i55(i55 [[BU1]], i55 [[BU2]])
  bu1 = __builtin_elementwise_add_sat(bu1, bu2);

  // CHECK:      [[IAS1:%.+]] = load i32, i32 addrspace(1)* @int_as_one, align 4
  // CHECK-NEXT: [[B:%.+]] = load i32, i32* @b, align 4
  // CHECK-NEXT: call i32 @llvm.sadd.sat.i32(i32 [[IAS1]], i32 [[B]])
  int_as_one = __builtin_elementwise_add_sat(int_as_one, b);

  // CHECK: call i32 @llvm.sadd.sat.i32(i32 1, i32 97)
  i1 = __builtin_elementwise_add_sat(1, 'a');
}

void test_builtin_elementwise_sub_sat(float f1, float f2, double d1, double d2,
                                      float4 vf1, float4 vf2, long long int i1,
                                      long long int i2, si8 vi1, si8 vi2,
                                      unsigned u1, unsigned u2, u4 vu1, u4 vu2,
                                      _BitInt(31) bi1, _BitInt(31) bi2,
                                      unsigned _BitInt(55) bu1, unsigned _BitInt(55) bu2) {
  // CHECK:      [[I1:%.+]] = load i64, i64* %i1.addr, align 8
  // CHECK-NEXT: [[I2:%.+]] = load i64, i64* %i2.addr, align 8
  // CHECK-NEXT: call i64 @llvm.ssub.sat.i64(i64 [[I1]], i64 [[I2]])
  i1 = __builtin_elementwise_sub_sat(i1, i2);

  // CHECK:      [[I1:%.+]] = load i64, i64* %i1.addr, align 8
  // CHECK-NEXT: call i64 @llvm.ssub.sat.i64(i64 [[I1]], i64 10)
  i1 = __builtin_elementwise_sub_sat(i1, 10);

  // CHECK:      [[VI1:%.+]] = load <8 x i16>, <8 x i16>* %vi1.addr, align 16
  // CHECK-NEXT: [[VI2:%.+]] = load <8 x i16>, <8 x i16>* %vi2.addr, align 16
  // CHECK-NEXT: call <8 x i16> @llvm.ssub.sat.v8i16(<8 x i16> [[VI1]], <8 x i16> [[VI2]])
  vi1 = __builtin_elementwise_sub_sat(vi1, vi2);

  // CHECK:      [[U1:%.+]] = load i32, i32* %u1.addr, align 4
  // CHECK-NEXT: [[U2:%.+]] = load i32, i32* %u2.addr, align 4
  // CHECK-NEXT: call i32 @llvm.usub.sat.i32(i32 [[U1]], i32 [[U2]])
  u1 = __builtin_elementwise_sub_sat(u1, u2);

  // CHECK:      [[VU1:%.+]] = load <4 x i32>, <4 x i32>* %vu1.addr, align 16
  // CHECK-NEXT: [[VU2:%.+]] = load <4 x i32>, <4 x i32>* %vu2.addr, align 16
  // CHECK-NEXT: call <4 x i32> @llvm.usub.sat.v4i32(<4 x i32> [[VU1]], <4 x i32> [[VU2]])
  vu1 = __builtin_elementwise_sub_sat(vu1, vu2);

  // CHECK:      [[BI1:%.+]] = load i31, i31* %bi1.addr, align 4
  // CHECK-NEXT: [[BI2:%.+]] = load i31, i31* %bi2.addr, align 4
  // CHECK-NEXT: call i31 @llvm.ssub.sat.i31(i31 [[BI1]], i31 [[BI2]])
  bi1 = __builtin_elementwise_sub_sat(bi1, bi2);

  // CHECK:      [[BU1:%.+]] = load i55, i55* %bu1.addr, align 8
  // CHECK-NEXT: [[BU2:%.+]] = load i55, i55* %bu2.addr, align 8
  // CHECK-NEXT: call i55 @llvm.usub.sat.i55(i55 [[BU1]], i55 [[BU2]])
  bu1 = __builtin_elementwise_sub_sat(bu1, bu2);

  // CHECK:      [[IAS1:%.+]] = load i32, i32 addrspace(1)* @int_as_one, align 4
  // CHECK-NEXT: [[B:%.+]] = load i32, i32* @b, align 4
  // CHECK-NEXT: call i32 @llvm.ssub.sat.i32(i32 [[IAS1]], i32 [[B]])
  int_as_one = __builtin_elementwise_sub_sat(int_as_one, b);

  // CHECK: call i32 @llvm.ssub.sat.i32(i32 1, i32 97)
  i1 = __builtin_elementwise_sub_sat(1, 'a');
}

void test_builtin_elementwise_max(float f1, float f2, double d1, double d2,
                                  float4 vf1, float4 vf2, long long int i1,
                                  long long int i2, si8 vi1, si8 vi2,
                                  unsigned u1, unsigned u2, u4 vu1, u4 vu2,
                                  _BitInt(31) bi1, _BitInt(31) bi2,
                                  unsigned _BitInt(55) bu1, unsigned _BitInt(55) bu2) {
  // CHECK-LABEL: define void @test_builtin_elementwise_max(
  // CHECK:      [[F1:%.+]] = load float, float* %f1.addr, align 4
  // CHECK-NEXT: [[F2:%.+]] = load float, float* %f2.addr, align 4
  // CHECK-NEXT:  call float @llvm.maxnum.f32(float %0, float %1)
  f1 = __builtin_elementwise_max(f1, f2);

  // CHECK:      [[D1:%.+]] = load double, double* %d1.addr, align 8
  // CHECK-NEXT: [[D2:%.+]] = load double, double* %d2.addr, align 8
  // CHECK-NEXT: call double @llvm.maxnum.f64(double [[D1]], double [[D2]])
  d1 = __builtin_elementwise_max(d1, d2);

  // CHECK:      [[D2:%.+]] = load double, double* %d2.addr, align 8
  // CHECK-NEXT: call double @llvm.maxnum.f64(double 2.000000e+01, double [[D2]])
  d1 = __builtin_elementwise_max(20.0, d2);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, <4 x float>* %vf1.addr, align 16
  // CHECK-NEXT: [[VF2:%.+]] = load <4 x float>, <4 x float>* %vf2.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.maxnum.v4f32(<4 x float> [[VF1]], <4 x float> [[VF2]])
  vf1 = __builtin_elementwise_max(vf1, vf2);

  // CHECK:      [[I1:%.+]] = load i64, i64* %i1.addr, align 8
  // CHECK-NEXT: [[I2:%.+]] = load i64, i64* %i2.addr, align 8
  // CHECK-NEXT: call i64 @llvm.smax.i64(i64 [[I1]], i64 [[I2]])
  i1 = __builtin_elementwise_max(i1, i2);

  // CHECK:      [[I1:%.+]] = load i64, i64* %i1.addr, align 8
  // CHECK-NEXT: call i64 @llvm.smax.i64(i64 [[I1]], i64 10)
  i1 = __builtin_elementwise_max(i1, 10);

  // CHECK:      [[VI1:%.+]] = load <8 x i16>, <8 x i16>* %vi1.addr, align 16
  // CHECK-NEXT: [[VI2:%.+]] = load <8 x i16>, <8 x i16>* %vi2.addr, align 16
  // CHECK-NEXT: call <8 x i16> @llvm.smax.v8i16(<8 x i16> [[VI1]], <8 x i16> [[VI2]])
  vi1 = __builtin_elementwise_max(vi1, vi2);

  // CHECK:      [[U1:%.+]] = load i32, i32* %u1.addr, align 4
  // CHECK-NEXT: [[U2:%.+]] = load i32, i32* %u2.addr, align 4
  // CHECK-NEXT: call i32 @llvm.umax.i32(i32 [[U1]], i32 [[U2]])
  u1 = __builtin_elementwise_max(u1, u2);

  // CHECK:      [[VU1:%.+]] = load <4 x i32>, <4 x i32>* %vu1.addr, align 16
  // CHECK-NEXT: [[VU2:%.+]] = load <4 x i32>, <4 x i32>* %vu2.addr, align 16
  // CHECK-NEXT: call <4 x i32> @llvm.umax.v4i32(<4 x i32> [[VU1]], <4 x i32> [[VU2]])
  vu1 = __builtin_elementwise_max(vu1, vu2);

  // CHECK:      [[BI1:%.+]] = load i31, i31* %bi1.addr, align 4
  // CHECK-NEXT: [[BI2:%.+]] = load i31, i31* %bi2.addr, align 4
  // CHECK-NEXT: call i31 @llvm.smax.i31(i31 [[BI1]], i31 [[BI2]])
  bi1 = __builtin_elementwise_max(bi1, bi2);

  // CHECK:      [[BU1:%.+]] = load i55, i55* %bu1.addr, align 8
  // CHECK-NEXT: [[BU2:%.+]] = load i55, i55* %bu2.addr, align 8
  // CHECK-NEXT: call i55 @llvm.umax.i55(i55 [[BU1]], i55 [[BU2]])
  bu1 = __builtin_elementwise_max(bu1, bu2);

  // CHECK:      [[CVF1:%.+]] = load <4 x float>, <4 x float>* %cvf1, align 16
  // CHECK-NEXT: [[VF2:%.+]] = load <4 x float>, <4 x float>* %vf2.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.maxnum.v4f32(<4 x float> [[CVF1]], <4 x float> [[VF2]])
  const float4 cvf1 = vf1;
  vf1 = __builtin_elementwise_max(cvf1, vf2);

  // CHECK:      [[VF2:%.+]] = load <4 x float>, <4 x float>* %vf2.addr, align 16
  // CHECK-NEXT: [[CVF1:%.+]] = load <4 x float>, <4 x float>* %cvf1, align 16
  // CHECK-NEXT: call <4 x float> @llvm.maxnum.v4f32(<4 x float> [[VF2]], <4 x float> [[CVF1]])
  vf1 = __builtin_elementwise_max(vf2, cvf1);

  // CHECK:      [[IAS1:%.+]] = load i32, i32 addrspace(1)* @int_as_one, align 4
  // CHECK-NEXT: [[B:%.+]] = load i32, i32* @b, align 4
  // CHECK-NEXT: call i32 @llvm.smax.i32(i32 [[IAS1]], i32 [[B]])
  int_as_one = __builtin_elementwise_max(int_as_one, b);

  // CHECK: call i32 @llvm.smax.i32(i32 1, i32 97)
  i1 = __builtin_elementwise_max(1, 'a');
}

void test_builtin_elementwise_min(float f1, float f2, double d1, double d2,
                                  float4 vf1, float4 vf2, long long int i1,
                                  long long int i2, si8 vi1, si8 vi2,
                                  unsigned u1, unsigned u2, u4 vu1, u4 vu2,
                                  _BitInt(31) bi1, _BitInt(31) bi2,
                                  unsigned _BitInt(55) bu1, unsigned _BitInt(55) bu2) {
  // CHECK-LABEL: define void @test_builtin_elementwise_min(
  // CHECK:      [[F1:%.+]] = load float, float* %f1.addr, align 4
  // CHECK-NEXT: [[F2:%.+]] = load float, float* %f2.addr, align 4
  // CHECK-NEXT:  call float @llvm.minnum.f32(float %0, float %1)
  f1 = __builtin_elementwise_min(f1, f2);

  // CHECK:      [[D1:%.+]] = load double, double* %d1.addr, align 8
  // CHECK-NEXT: [[D2:%.+]] = load double, double* %d2.addr, align 8
  // CHECK-NEXT: call double @llvm.minnum.f64(double [[D1]], double [[D2]])
  d1 = __builtin_elementwise_min(d1, d2);

  // CHECK:      [[D1:%.+]] = load double, double* %d1.addr, align 8
  // CHECK-NEXT: call double @llvm.minnum.f64(double [[D1]], double 2.000000e+00)
  d1 = __builtin_elementwise_min(d1, 2.0);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, <4 x float>* %vf1.addr, align 16
  // CHECK-NEXT: [[VF2:%.+]] = load <4 x float>, <4 x float>* %vf2.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.minnum.v4f32(<4 x float> [[VF1]], <4 x float> [[VF2]])
  vf1 = __builtin_elementwise_min(vf1, vf2);

  // CHECK:      [[I1:%.+]] = load i64, i64* %i1.addr, align 8
  // CHECK-NEXT: [[I2:%.+]] = load i64, i64* %i2.addr, align 8
  // CHECK-NEXT: call i64 @llvm.smin.i64(i64 [[I1]], i64 [[I2]])
  i1 = __builtin_elementwise_min(i1, i2);

  // CHECK:      [[I2:%.+]] = load i64, i64* %i2.addr, align 8
  // CHECK-NEXT: call i64 @llvm.smin.i64(i64 -11, i64 [[I2]])
  i1 = __builtin_elementwise_min(-11, i2);

  // CHECK:      [[VI1:%.+]] = load <8 x i16>, <8 x i16>* %vi1.addr, align 16
  // CHECK-NEXT: [[VI2:%.+]] = load <8 x i16>, <8 x i16>* %vi2.addr, align 16
  // CHECK-NEXT: call <8 x i16> @llvm.smin.v8i16(<8 x i16> [[VI1]], <8 x i16> [[VI2]])
  vi1 = __builtin_elementwise_min(vi1, vi2);

  // CHECK:      [[U1:%.+]] = load i32, i32* %u1.addr, align 4
  // CHECK-NEXT: [[U2:%.+]] = load i32, i32* %u2.addr, align 4
  // CHECK-NEXT: call i32 @llvm.umin.i32(i32 [[U1]], i32 [[U2]])
  u1 = __builtin_elementwise_min(u1, u2);

  // CHECK:      [[U1:%.+]] = load i32, i32* %u1.addr, align 4
  // CHECK-NEXT: [[ZEXT_U1:%.+]] = zext i32 [[U1]] to i64
  // CHECK-NEXT: [[I2:%.+]] = load i64, i64* %i2.addr, align 8
  // CHECK-NEXT: call i64 @llvm.smin.i64(i64 [[ZEXT_U1]], i64 [[I2]])
  u1 = __builtin_elementwise_min(u1, i2);

  // CHECK:      [[VU1:%.+]] = load <4 x i32>, <4 x i32>* %vu1.addr, align 16
  // CHECK-NEXT: [[VU2:%.+]] = load <4 x i32>, <4 x i32>* %vu2.addr, align 16
  // CHECK-NEXT: call <4 x i32> @llvm.umin.v4i32(<4 x i32> [[VU1]], <4 x i32> [[VU2]])
  vu1 = __builtin_elementwise_min(vu1, vu2);

  // CHECK:      [[BI1:%.+]] = load i31, i31* %bi1.addr, align 4
  // CHECK-NEXT: [[BI2:%.+]] = load i31, i31* %bi2.addr, align 4
  // CHECK-NEXT: call i31 @llvm.smin.i31(i31 [[BI1]], i31 [[BI2]])
  bi1 = __builtin_elementwise_min(bi1, bi2);

  // CHECK:      [[BU1:%.+]] = load i55, i55* %bu1.addr, align 8
  // CHECK-NEXT: [[BU2:%.+]] = load i55, i55* %bu2.addr, align 8
  // CHECK-NEXT: call i55 @llvm.umin.i55(i55 [[BU1]], i55 [[BU2]])
  bu1 = __builtin_elementwise_min(bu1, bu2);

  // CHECK:      [[CVF1:%.+]] = load <4 x float>, <4 x float>* %cvf1, align 16
  // CHECK-NEXT: [[VF2:%.+]] = load <4 x float>, <4 x float>* %vf2.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.minnum.v4f32(<4 x float> [[CVF1]], <4 x float> [[VF2]])
  const float4 cvf1 = vf1;
  vf1 = __builtin_elementwise_min(cvf1, vf2);

  // CHECK:      [[VF2:%.+]] = load <4 x float>, <4 x float>* %vf2.addr, align 16
  // CHECK-NEXT: [[CVF1:%.+]] = load <4 x float>, <4 x float>* %cvf1, align 16
  // CHECK-NEXT: call <4 x float> @llvm.minnum.v4f32(<4 x float> [[VF2]], <4 x float> [[CVF1]])
  vf1 = __builtin_elementwise_min(vf2, cvf1);

  // CHECK:      [[IAS1:%.+]] = load i32, i32 addrspace(1)* @int_as_one, align 4
  // CHECK-NEXT: [[B:%.+]] = load i32, i32* @b, align 4
  // CHECK-NEXT: call i32 @llvm.smin.i32(i32 [[IAS1]], i32 [[B]])
  int_as_one = __builtin_elementwise_min(int_as_one, b);
}

void test_builtin_elementwise_ceil(float f1, float f2, double d1, double d2,
                                   float4 vf1, float4 vf2) {
  // CHECK-LABEL: define void @test_builtin_elementwise_ceil(
  // CHECK:      [[F1:%.+]] = load float, float* %f1.addr, align 4
  // CHECK-NEXT:  call float @llvm.ceil.f32(float [[F1]])
  f2 = __builtin_elementwise_ceil(f1);

  // CHECK:      [[D1:%.+]] = load double, double* %d1.addr, align 8
  // CHECK-NEXT: call double @llvm.ceil.f64(double [[D1]])
  d2 = __builtin_elementwise_ceil(d1);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, <4 x float>* %vf1.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.ceil.v4f32(<4 x float> [[VF1]])
  vf2 = __builtin_elementwise_ceil(vf1);
}

void test_builtin_elementwise_floor(float f1, float f2, double d1, double d2,
                                    float4 vf1, float4 vf2) {
  // CHECK-LABEL: define void @test_builtin_elementwise_floor(
  // CHECK:      [[F1:%.+]] = load float, float* %f1.addr, align 4
  // CHECK-NEXT:  call float @llvm.floor.f32(float [[F1]])
  f2 = __builtin_elementwise_floor(f1);

  // CHECK:      [[D1:%.+]] = load double, double* %d1.addr, align 8
  // CHECK-NEXT: call double @llvm.floor.f64(double [[D1]])
  d2 = __builtin_elementwise_floor(d1);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, <4 x float>* %vf1.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.floor.v4f32(<4 x float> [[VF1]])
  vf2 = __builtin_elementwise_floor(vf1);
}

void test_builtin_elementwise_roundeven(float f1, float f2, double d1, double d2,
                                        float4 vf1, float4 vf2) {
  // CHECK-LABEL: define void @test_builtin_elementwise_roundeven(
  // CHECK:      [[F1:%.+]] = load float, float* %f1.addr, align 4
  // CHECK-NEXT:  call float @llvm.roundeven.f32(float [[F1]])
  f2 = __builtin_elementwise_roundeven(f1);

  // CHECK:      [[D1:%.+]] = load double, double* %d1.addr, align 8
  // CHECK-NEXT: call double @llvm.roundeven.f64(double [[D1]])
  d2 = __builtin_elementwise_roundeven(d1);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, <4 x float>* %vf1.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.roundeven.v4f32(<4 x float> [[VF1]])
  vf2 = __builtin_elementwise_roundeven(vf1);
}

void test_builtin_elementwise_trunc(float f1, float f2, double d1, double d2,
                                    float4 vf1, float4 vf2) {
  // CHECK-LABEL: define void @test_builtin_elementwise_trunc(
  // CHECK:      [[F1:%.+]] = load float, float* %f1.addr, align 4
  // CHECK-NEXT:  call float @llvm.trunc.f32(float [[F1]])
  f2 = __builtin_elementwise_trunc(f1);

  // CHECK:      [[D1:%.+]] = load double, double* %d1.addr, align 8
  // CHECK-NEXT: call double @llvm.trunc.f64(double [[D1]])
  d2 = __builtin_elementwise_trunc(d1);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, <4 x float>* %vf1.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.trunc.v4f32(<4 x float> [[VF1]])
  vf2 = __builtin_elementwise_trunc(vf1);
}
