// RUN: %clang_cc1 -triple x86_64-apple-darwin %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s

typedef _Float16 half;

typedef half half2 __attribute__((ext_vector_type(2)));
typedef float float2 __attribute__((ext_vector_type(2)));
typedef float float4 __attribute__((ext_vector_type(4)));
typedef short int si8 __attribute__((ext_vector_type(8)));
typedef unsigned int u4 __attribute__((ext_vector_type(4)));
typedef double double2 __attribute__((ext_vector_type(2)));
typedef double double3 __attribute__((ext_vector_type(3)));

__attribute__((address_space(1))) int int_as_one;
typedef int bar;
bar b;

void test_builtin_elementwise_abs(float f1, float f2, double d1, double d2,
                                  float4 vf1, float4 vf2, si8 vi1, si8 vi2,
                                  long long int i1, long long int i2, short si,
                                  _BitInt(31) bi1, _BitInt(31) bi2) {
  // CHECK-LABEL: define void @test_builtin_elementwise_abs(
  // CHECK:      [[F1:%.+]] = load float, ptr %f1.addr, align 4
  // CHECK-NEXT:  call float @llvm.fabs.f32(float [[F1]])
  f2 = __builtin_elementwise_abs(f1);

  // CHECK:      [[D1:%.+]] = load double, ptr %d1.addr, align 8
  // CHECK-NEXT: call double @llvm.fabs.f64(double [[D1]])
  d2 = __builtin_elementwise_abs(d1);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, ptr %vf1.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.fabs.v4f32(<4 x float> [[VF1]])
  vf2 = __builtin_elementwise_abs(vf1);

  // CHECK:      [[I1:%.+]] = load i64, ptr %i1.addr, align 8
  // CHECK-NEXT: call i64 @llvm.abs.i64(i64 [[I1]], i1 false)
  i2 = __builtin_elementwise_abs(i1);

  // CHECK:      [[VI1:%.+]] = load <8 x i16>, ptr %vi1.addr, align 16
  // CHECK-NEXT: call <8 x i16> @llvm.abs.v8i16(<8 x i16> [[VI1]], i1 false)
  vi2 = __builtin_elementwise_abs(vi1);

  // CHECK:      [[CVI2:%.+]] = load <8 x i16>, ptr %cvi2, align 16
  // CHECK-NEXT: call <8 x i16> @llvm.abs.v8i16(<8 x i16> [[CVI2]], i1 false)
  const si8 cvi2 = vi2;
  vi2 = __builtin_elementwise_abs(cvi2);

  // CHECK:      [[BI1:%.+]] = load i32, ptr %bi1.addr, align 4
  // CHECK-NEXT: [[LOADEDV:%.+]] = trunc i32 [[BI1]] to i31
  // CHECK-NEXT: call i31 @llvm.abs.i31(i31 [[LOADEDV]], i1 false)
  bi2 = __builtin_elementwise_abs(bi1);

  // CHECK:      [[IA1:%.+]] = load i32, ptr addrspace(1) @int_as_one, align 4
  // CHECK-NEXT: call i32 @llvm.abs.i32(i32 [[IA1]], i1 false)
  b = __builtin_elementwise_abs(int_as_one);

  // CHECK:   call i32 @llvm.abs.i32(i32 -10, i1 false)
  b = __builtin_elementwise_abs(-10);

  // CHECK:      [[SI:%.+]] = load i16, ptr %si.addr, align 2
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
  // CHECK:      [[I1:%.+]] = load i64, ptr %i1.addr, align 8
  // CHECK-NEXT: [[I2:%.+]] = load i64, ptr %i2.addr, align 8
  // CHECK-NEXT: call i64 @llvm.sadd.sat.i64(i64 [[I1]], i64 [[I2]])
  i1 = __builtin_elementwise_add_sat(i1, i2);

  // CHECK:      [[I1:%.+]] = load i64, ptr %i1.addr, align 8
  // CHECK-NEXT: call i64 @llvm.sadd.sat.i64(i64 [[I1]], i64 10)
  i1 = __builtin_elementwise_add_sat(i1, 10);

  // CHECK:      [[VI1:%.+]] = load <8 x i16>, ptr %vi1.addr, align 16
  // CHECK-NEXT: [[VI2:%.+]] = load <8 x i16>, ptr %vi2.addr, align 16
  // CHECK-NEXT: call <8 x i16> @llvm.sadd.sat.v8i16(<8 x i16> [[VI1]], <8 x i16> [[VI2]])
  vi1 = __builtin_elementwise_add_sat(vi1, vi2);

  // CHECK:      [[U1:%.+]] = load i32, ptr %u1.addr, align 4
  // CHECK-NEXT: [[U2:%.+]] = load i32, ptr %u2.addr, align 4
  // CHECK-NEXT: call i32 @llvm.uadd.sat.i32(i32 [[U1]], i32 [[U2]])
  u1 = __builtin_elementwise_add_sat(u1, u2);

  // CHECK:      [[VU1:%.+]] = load <4 x i32>, ptr %vu1.addr, align 16
  // CHECK-NEXT: [[VU2:%.+]] = load <4 x i32>, ptr %vu2.addr, align 16
  // CHECK-NEXT: call <4 x i32> @llvm.uadd.sat.v4i32(<4 x i32> [[VU1]], <4 x i32> [[VU2]])
  vu1 = __builtin_elementwise_add_sat(vu1, vu2);

  // CHECK:      [[BI1:%.+]] = load i32, ptr %bi1.addr, align 4
  // CHECK-NEXT: [[LOADEDV:%.+]] = trunc i32 [[BI1]] to i31
  // CHECK-NEXT: [[BI2:%.+]] = load i32, ptr %bi2.addr, align 4
  // CHECK-NEXT: [[LOADEDV1:%.+]] = trunc i32 [[BI2]] to i31
  // CHECK-NEXT: call i31 @llvm.sadd.sat.i31(i31 [[LOADEDV]], i31 [[LOADEDV1]])
  bi1 = __builtin_elementwise_add_sat(bi1, bi2);

  // CHECK:      [[BU1:%.+]] = load i64, ptr %bu1.addr, align 8
  // CHECK-NEXT: [[LOADEDV2:%.+]] = trunc i64 [[BU1]] to i55
  // CHECK-NEXT: [[BU2:%.+]] = load i64, ptr %bu2.addr, align 8
  // CHECK-NEXT: [[LOADEDV3:%.+]] = trunc i64 [[BU2]] to i55
  // CHECK-NEXT: call i55 @llvm.uadd.sat.i55(i55 [[LOADEDV2]], i55 [[LOADEDV3]])
  bu1 = __builtin_elementwise_add_sat(bu1, bu2);

  // CHECK:      [[IAS1:%.+]] = load i32, ptr addrspace(1) @int_as_one, align 4
  // CHECK-NEXT: [[B:%.+]] = load i32, ptr @b, align 4
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
  // CHECK:      [[I1:%.+]] = load i64, ptr %i1.addr, align 8
  // CHECK-NEXT: [[I2:%.+]] = load i64, ptr %i2.addr, align 8
  // CHECK-NEXT: call i64 @llvm.ssub.sat.i64(i64 [[I1]], i64 [[I2]])
  i1 = __builtin_elementwise_sub_sat(i1, i2);

  // CHECK:      [[I1:%.+]] = load i64, ptr %i1.addr, align 8
  // CHECK-NEXT: call i64 @llvm.ssub.sat.i64(i64 [[I1]], i64 10)
  i1 = __builtin_elementwise_sub_sat(i1, 10);

  // CHECK:      [[VI1:%.+]] = load <8 x i16>, ptr %vi1.addr, align 16
  // CHECK-NEXT: [[VI2:%.+]] = load <8 x i16>, ptr %vi2.addr, align 16
  // CHECK-NEXT: call <8 x i16> @llvm.ssub.sat.v8i16(<8 x i16> [[VI1]], <8 x i16> [[VI2]])
  vi1 = __builtin_elementwise_sub_sat(vi1, vi2);

  // CHECK:      [[U1:%.+]] = load i32, ptr %u1.addr, align 4
  // CHECK-NEXT: [[U2:%.+]] = load i32, ptr %u2.addr, align 4
  // CHECK-NEXT: call i32 @llvm.usub.sat.i32(i32 [[U1]], i32 [[U2]])
  u1 = __builtin_elementwise_sub_sat(u1, u2);

  // CHECK:      [[VU1:%.+]] = load <4 x i32>, ptr %vu1.addr, align 16
  // CHECK-NEXT: [[VU2:%.+]] = load <4 x i32>, ptr %vu2.addr, align 16
  // CHECK-NEXT: call <4 x i32> @llvm.usub.sat.v4i32(<4 x i32> [[VU1]], <4 x i32> [[VU2]])
  vu1 = __builtin_elementwise_sub_sat(vu1, vu2);

  // CHECK:      [[BI1:%.+]] = load i32, ptr %bi1.addr, align 4
  // CHECK-NEXT: [[LOADEDV:%.+]] = trunc i32 [[BI1]] to i31
  // CHECK-NEXT: [[BI2:%.+]] = load i32, ptr %bi2.addr, align 4
  // CHECK-NEXT: [[LOADEDV1:%.+]] = trunc i32 [[BI2]] to i31
  // CHECK-NEXT: call i31 @llvm.ssub.sat.i31(i31 [[LOADEDV]], i31 [[LOADEDV1]])
  bi1 = __builtin_elementwise_sub_sat(bi1, bi2);

  // CHECK:      [[BU1:%.+]] = load i64, ptr %bu1.addr, align 8
  // CHECK-NEXT: [[LOADEDV2:%.+]] = trunc i64 [[BU1]] to i55
  // CHECK-NEXT: [[BU2:%.+]] = load i64, ptr %bu2.addr, align 8
  // CHECK-NEXT: [[LOADEDV3:%.+]] = trunc i64 [[BU2]] to i55
  // CHECK-NEXT: call i55 @llvm.usub.sat.i55(i55 [[LOADEDV2]], i55 [[LOADEDV3]])
  bu1 = __builtin_elementwise_sub_sat(bu1, bu2);

  // CHECK:      [[IAS1:%.+]] = load i32, ptr addrspace(1) @int_as_one, align 4
  // CHECK-NEXT: [[B:%.+]] = load i32, ptr @b, align 4
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
  // CHECK:      [[F1:%.+]] = load float, ptr %f1.addr, align 4
  // CHECK-NEXT: [[F2:%.+]] = load float, ptr %f2.addr, align 4
  // CHECK-NEXT:  call float @llvm.maxnum.f32(float [[F1]], float [[F2]])
  f1 = __builtin_elementwise_max(f1, f2);

  // CHECK:      [[D1:%.+]] = load double, ptr %d1.addr, align 8
  // CHECK-NEXT: [[D2:%.+]] = load double, ptr %d2.addr, align 8
  // CHECK-NEXT: call double @llvm.maxnum.f64(double [[D1]], double [[D2]])
  d1 = __builtin_elementwise_max(d1, d2);

  // CHECK:      [[D2:%.+]] = load double, ptr %d2.addr, align 8
  // CHECK-NEXT: call double @llvm.maxnum.f64(double 2.000000e+01, double [[D2]])
  d1 = __builtin_elementwise_max(20.0, d2);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, ptr %vf1.addr, align 16
  // CHECK-NEXT: [[VF2:%.+]] = load <4 x float>, ptr %vf2.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.maxnum.v4f32(<4 x float> [[VF1]], <4 x float> [[VF2]])
  vf1 = __builtin_elementwise_max(vf1, vf2);

  // CHECK:      [[I1:%.+]] = load i64, ptr %i1.addr, align 8
  // CHECK-NEXT: [[I2:%.+]] = load i64, ptr %i2.addr, align 8
  // CHECK-NEXT: call i64 @llvm.smax.i64(i64 [[I1]], i64 [[I2]])
  i1 = __builtin_elementwise_max(i1, i2);

  // CHECK:      [[I1:%.+]] = load i64, ptr %i1.addr, align 8
  // CHECK-NEXT: call i64 @llvm.smax.i64(i64 [[I1]], i64 10)
  i1 = __builtin_elementwise_max(i1, 10);

  // CHECK:      [[VI1:%.+]] = load <8 x i16>, ptr %vi1.addr, align 16
  // CHECK-NEXT: [[VI2:%.+]] = load <8 x i16>, ptr %vi2.addr, align 16
  // CHECK-NEXT: call <8 x i16> @llvm.smax.v8i16(<8 x i16> [[VI1]], <8 x i16> [[VI2]])
  vi1 = __builtin_elementwise_max(vi1, vi2);

  // CHECK:      [[U1:%.+]] = load i32, ptr %u1.addr, align 4
  // CHECK-NEXT: [[U2:%.+]] = load i32, ptr %u2.addr, align 4
  // CHECK-NEXT: call i32 @llvm.umax.i32(i32 [[U1]], i32 [[U2]])
  u1 = __builtin_elementwise_max(u1, u2);

  // CHECK:      [[VU1:%.+]] = load <4 x i32>, ptr %vu1.addr, align 16
  // CHECK-NEXT: [[VU2:%.+]] = load <4 x i32>, ptr %vu2.addr, align 16
  // CHECK-NEXT: call <4 x i32> @llvm.umax.v4i32(<4 x i32> [[VU1]], <4 x i32> [[VU2]])
  vu1 = __builtin_elementwise_max(vu1, vu2);

  // CHECK:      [[BI1:%.+]] = load i32, ptr %bi1.addr, align 4
  // CHECK-NEXT: [[LOADEDV:%.+]] = trunc i32 [[BI1]] to i31
  // CHECK-NEXT: [[BI2:%.+]] = load i32, ptr %bi2.addr, align 4
  // CHECK-NEXT: [[LOADEDV1:%.+]] = trunc i32 [[BI2]] to i31
  // CHECK-NEXT: call i31 @llvm.smax.i31(i31 [[LOADEDV]], i31 [[LOADEDV1]])
  bi1 = __builtin_elementwise_max(bi1, bi2);

  // CHECK:      [[BU1:%.+]] = load i64, ptr %bu1.addr, align 8
  // CHECK-NEXT: [[LOADEDV2:%.+]] = trunc i64 [[BU1]] to i55
  // CHECK-NEXT: [[BU2:%.+]] = load i64, ptr %bu2.addr, align 8
  // CHECK-NEXT: [[LOADEDV3:%.+]] = trunc i64 [[BU2]] to i55
  // CHECK-NEXT: call i55 @llvm.umax.i55(i55 [[LOADEDV2]], i55 [[LOADEDV3]])
  bu1 = __builtin_elementwise_max(bu1, bu2);

  // CHECK:      [[CVF1:%.+]] = load <4 x float>, ptr %cvf1, align 16
  // CHECK-NEXT: [[VF2:%.+]] = load <4 x float>, ptr %vf2.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.maxnum.v4f32(<4 x float> [[CVF1]], <4 x float> [[VF2]])
  const float4 cvf1 = vf1;
  vf1 = __builtin_elementwise_max(cvf1, vf2);

  // CHECK:      [[VF2:%.+]] = load <4 x float>, ptr %vf2.addr, align 16
  // CHECK-NEXT: [[CVF1:%.+]] = load <4 x float>, ptr %cvf1, align 16
  // CHECK-NEXT: call <4 x float> @llvm.maxnum.v4f32(<4 x float> [[VF2]], <4 x float> [[CVF1]])
  vf1 = __builtin_elementwise_max(vf2, cvf1);

  // CHECK:      [[IAS1:%.+]] = load i32, ptr addrspace(1) @int_as_one, align 4
  // CHECK-NEXT: [[B:%.+]] = load i32, ptr @b, align 4
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
  // CHECK:      [[F1:%.+]] = load float, ptr %f1.addr, align 4
  // CHECK-NEXT: [[F2:%.+]] = load float, ptr %f2.addr, align 4
  // CHECK-NEXT:  call float @llvm.minnum.f32(float [[F1]], float [[F2]])
  f1 = __builtin_elementwise_min(f1, f2);

  // CHECK:      [[D1:%.+]] = load double, ptr %d1.addr, align 8
  // CHECK-NEXT: [[D2:%.+]] = load double, ptr %d2.addr, align 8
  // CHECK-NEXT: call double @llvm.minnum.f64(double [[D1]], double [[D2]])
  d1 = __builtin_elementwise_min(d1, d2);

  // CHECK:      [[D1:%.+]] = load double, ptr %d1.addr, align 8
  // CHECK-NEXT: call double @llvm.minnum.f64(double [[D1]], double 2.000000e+00)
  d1 = __builtin_elementwise_min(d1, 2.0);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, ptr %vf1.addr, align 16
  // CHECK-NEXT: [[VF2:%.+]] = load <4 x float>, ptr %vf2.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.minnum.v4f32(<4 x float> [[VF1]], <4 x float> [[VF2]])
  vf1 = __builtin_elementwise_min(vf1, vf2);

  // CHECK:      [[I1:%.+]] = load i64, ptr %i1.addr, align 8
  // CHECK-NEXT: [[I2:%.+]] = load i64, ptr %i2.addr, align 8
  // CHECK-NEXT: call i64 @llvm.smin.i64(i64 [[I1]], i64 [[I2]])
  i1 = __builtin_elementwise_min(i1, i2);

  // CHECK:      [[I2:%.+]] = load i64, ptr %i2.addr, align 8
  // CHECK-NEXT: call i64 @llvm.smin.i64(i64 -11, i64 [[I2]])
  i1 = __builtin_elementwise_min(-11, i2);

  // CHECK:      [[VI1:%.+]] = load <8 x i16>, ptr %vi1.addr, align 16
  // CHECK-NEXT: [[VI2:%.+]] = load <8 x i16>, ptr %vi2.addr, align 16
  // CHECK-NEXT: call <8 x i16> @llvm.smin.v8i16(<8 x i16> [[VI1]], <8 x i16> [[VI2]])
  vi1 = __builtin_elementwise_min(vi1, vi2);

  // CHECK:      [[U1:%.+]] = load i32, ptr %u1.addr, align 4
  // CHECK-NEXT: [[U2:%.+]] = load i32, ptr %u2.addr, align 4
  // CHECK-NEXT: call i32 @llvm.umin.i32(i32 [[U1]], i32 [[U2]])
  u1 = __builtin_elementwise_min(u1, u2);

  // CHECK:      [[U1:%.+]] = load i32, ptr %u1.addr, align 4
  // CHECK-NEXT: [[ZEXT_U1:%.+]] = zext i32 [[U1]] to i64
  // CHECK-NEXT: [[I2:%.+]] = load i64, ptr %i2.addr, align 8
  // CHECK-NEXT: call i64 @llvm.smin.i64(i64 [[ZEXT_U1]], i64 [[I2]])
  u1 = __builtin_elementwise_min(u1, i2);

  // CHECK:      [[VU1:%.+]] = load <4 x i32>, ptr %vu1.addr, align 16
  // CHECK-NEXT: [[VU2:%.+]] = load <4 x i32>, ptr %vu2.addr, align 16
  // CHECK-NEXT: call <4 x i32> @llvm.umin.v4i32(<4 x i32> [[VU1]], <4 x i32> [[VU2]])
  vu1 = __builtin_elementwise_min(vu1, vu2);

  // CHECK:      [[BI1:%.+]] = load i32, ptr %bi1.addr, align 4
  // CHECK-NEXT: [[LOADEDV:%.+]] = trunc i32 [[BI1]] to i31
  // CHECK-NEXT: [[BI2:%.+]] = load i32, ptr %bi2.addr, align 4
  // CHECK-NEXT: [[LOADEDV1:%.+]] = trunc i32 [[BI2]] to i31
  // CHECK-NEXT: call i31 @llvm.smin.i31(i31 [[LOADEDV]], i31 [[LOADEDV1]])
  bi1 = __builtin_elementwise_min(bi1, bi2);

  // CHECK:      [[BU1:%.+]] = load i64, ptr %bu1.addr, align 8
  // CHECK-NEXT: [[LOADEDV2:%.+]] = trunc i64 [[BU1]] to i55
  // CHECK-NEXT: [[BU2:%.+]] = load i64, ptr %bu2.addr, align 8
  // CHECK-NEXT: [[LOADEDV3:%.+]] = trunc i64 [[BU2]] to i55
  // CHECK-NEXT: call i55 @llvm.umin.i55(i55 [[LOADEDV2]], i55 [[LOADEDV3]])
  bu1 = __builtin_elementwise_min(bu1, bu2);

  // CHECK:      [[CVF1:%.+]] = load <4 x float>, ptr %cvf1, align 16
  // CHECK-NEXT: [[VF2:%.+]] = load <4 x float>, ptr %vf2.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.minnum.v4f32(<4 x float> [[CVF1]], <4 x float> [[VF2]])
  const float4 cvf1 = vf1;
  vf1 = __builtin_elementwise_min(cvf1, vf2);

  // CHECK:      [[VF2:%.+]] = load <4 x float>, ptr %vf2.addr, align 16
  // CHECK-NEXT: [[CVF1:%.+]] = load <4 x float>, ptr %cvf1, align 16
  // CHECK-NEXT: call <4 x float> @llvm.minnum.v4f32(<4 x float> [[VF2]], <4 x float> [[CVF1]])
  vf1 = __builtin_elementwise_min(vf2, cvf1);

  // CHECK:      [[IAS1:%.+]] = load i32, ptr addrspace(1) @int_as_one, align 4
  // CHECK-NEXT: [[B:%.+]] = load i32, ptr @b, align 4
  // CHECK-NEXT: call i32 @llvm.smin.i32(i32 [[IAS1]], i32 [[B]])
  int_as_one = __builtin_elementwise_min(int_as_one, b);
}

void test_builtin_elementwise_bitreverse(si8 vi1, si8 vi2,
                                  long long int i1, long long int i2, short si,
                                  _BitInt(31) bi1, _BitInt(31) bi2) {
  

  // CHECK:      [[I1:%.+]] = load i64, ptr %i1.addr, align 8
  // CHECK-NEXT: call i64 @llvm.bitreverse.i64(i64 [[I1]])
  i2 = __builtin_elementwise_bitreverse(i1);

  // CHECK:      [[VI1:%.+]] = load <8 x i16>, ptr %vi1.addr, align 16
  // CHECK-NEXT: call <8 x i16> @llvm.bitreverse.v8i16(<8 x i16> [[VI1]])
  vi2 = __builtin_elementwise_bitreverse(vi1);

  // CHECK:      [[CVI2:%.+]] = load <8 x i16>, ptr %cvi2, align 16
  // CHECK-NEXT: call <8 x i16> @llvm.bitreverse.v8i16(<8 x i16> [[CVI2]])
  const si8 cvi2 = vi2;
  vi2 = __builtin_elementwise_bitreverse(cvi2);

  // CHECK:      [[BI1:%.+]] = load i32, ptr %bi1.addr, align 4
  // CHECK-NEXT: [[LOADEDV:%.+]] = trunc i32 [[BI1]] to i31
  // CHECK-NEXT: call i31 @llvm.bitreverse.i31(i31 [[LOADEDV]])
  bi2 = __builtin_elementwise_bitreverse(bi1);

  // CHECK:      [[IA1:%.+]] = load i32, ptr addrspace(1) @int_as_one, align 4
  // CHECK-NEXT: call i32 @llvm.bitreverse.i32(i32 [[IA1]])
  b = __builtin_elementwise_bitreverse(int_as_one);

  // CHECK:   call i32 @llvm.bitreverse.i32(i32 -10)
  b = __builtin_elementwise_bitreverse(-10);

  // CHECK:      [[SI:%.+]] = load i16, ptr %si.addr, align 2
  // CHECK-NEXT: [[SI_EXT:%.+]] = sext i16 [[SI]] to i32
  // CHECK-NEXT: [[RES:%.+]] = call i32 @llvm.bitreverse.i32(i32 [[SI_EXT]])
  // CHECK-NEXT: = trunc i32 [[RES]] to i16
  si = __builtin_elementwise_bitreverse(si);
}

void test_builtin_elementwise_ceil(float f1, float f2, double d1, double d2,
                                   float4 vf1, float4 vf2) {
  // CHECK-LABEL: define void @test_builtin_elementwise_ceil(
  // CHECK:      [[F1:%.+]] = load float, ptr %f1.addr, align 4
  // CHECK-NEXT:  call float @llvm.ceil.f32(float [[F1]])
  f2 = __builtin_elementwise_ceil(f1);

  // CHECK:      [[D1:%.+]] = load double, ptr %d1.addr, align 8
  // CHECK-NEXT: call double @llvm.ceil.f64(double [[D1]])
  d2 = __builtin_elementwise_ceil(d1);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, ptr %vf1.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.ceil.v4f32(<4 x float> [[VF1]])
  vf2 = __builtin_elementwise_ceil(vf1);
}

void test_builtin_elementwise_acos(float f1, float f2, double d1, double d2,
                                  float4 vf1, float4 vf2) {
  // CHECK-LABEL: define void @test_builtin_elementwise_acos(
  // CHECK:      [[F1:%.+]] = load float, ptr %f1.addr, align 4
  // CHECK-NEXT:  call float @llvm.acos.f32(float [[F1]])
  f2 = __builtin_elementwise_acos(f1);

  // CHECK:      [[D1:%.+]] = load double, ptr %d1.addr, align 8
  // CHECK-NEXT: call double @llvm.acos.f64(double [[D1]])
  d2 = __builtin_elementwise_acos(d1);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, ptr %vf1.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.acos.v4f32(<4 x float> [[VF1]])
  vf2 = __builtin_elementwise_acos(vf1);
}

void test_builtin_elementwise_asin(float f1, float f2, double d1, double d2,
                                  float4 vf1, float4 vf2) {
  // CHECK-LABEL: define void @test_builtin_elementwise_asin(
  // CHECK:      [[F1:%.+]] = load float, ptr %f1.addr, align 4
  // CHECK-NEXT:  call float @llvm.asin.f32(float [[F1]])
  f2 = __builtin_elementwise_asin(f1);

  // CHECK:      [[D1:%.+]] = load double, ptr %d1.addr, align 8
  // CHECK-NEXT: call double @llvm.asin.f64(double [[D1]])
  d2 = __builtin_elementwise_asin(d1);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, ptr %vf1.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.asin.v4f32(<4 x float> [[VF1]])
  vf2 = __builtin_elementwise_asin(vf1);
}

void test_builtin_elementwise_atan(float f1, float f2, double d1, double d2,
                                  float4 vf1, float4 vf2) {
  // CHECK-LABEL: define void @test_builtin_elementwise_atan(
  // CHECK:      [[F1:%.+]] = load float, ptr %f1.addr, align 4
  // CHECK-NEXT:  call float @llvm.atan.f32(float [[F1]])
  f2 = __builtin_elementwise_atan(f1);

  // CHECK:      [[D1:%.+]] = load double, ptr %d1.addr, align 8
  // CHECK-NEXT: call double @llvm.atan.f64(double [[D1]])
  d2 = __builtin_elementwise_atan(d1);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, ptr %vf1.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.atan.v4f32(<4 x float> [[VF1]])
  vf2 = __builtin_elementwise_atan(vf1);
}

void test_builtin_elementwise_cos(float f1, float f2, double d1, double d2,
                                  float4 vf1, float4 vf2) {
  // CHECK-LABEL: define void @test_builtin_elementwise_cos(
  // CHECK:      [[F1:%.+]] = load float, ptr %f1.addr, align 4
  // CHECK-NEXT:  call float @llvm.cos.f32(float [[F1]])
  f2 = __builtin_elementwise_cos(f1);

  // CHECK:      [[D1:%.+]] = load double, ptr %d1.addr, align 8
  // CHECK-NEXT: call double @llvm.cos.f64(double [[D1]])
  d2 = __builtin_elementwise_cos(d1);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, ptr %vf1.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.cos.v4f32(<4 x float> [[VF1]])
  vf2 = __builtin_elementwise_cos(vf1);
}

void test_builtin_elementwise_cosh(float f1, float f2, double d1, double d2,
                                  float4 vf1, float4 vf2) {
  // CHECK-LABEL: define void @test_builtin_elementwise_cosh(
  // CHECK:      [[F1:%.+]] = load float, ptr %f1.addr, align 4
  // CHECK-NEXT:  call float @llvm.cosh.f32(float [[F1]])
  f2 = __builtin_elementwise_cosh(f1);

  // CHECK:      [[D1:%.+]] = load double, ptr %d1.addr, align 8
  // CHECK-NEXT: call double @llvm.cosh.f64(double [[D1]])
  d2 = __builtin_elementwise_cosh(d1);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, ptr %vf1.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.cosh.v4f32(<4 x float> [[VF1]])
  vf2 = __builtin_elementwise_cosh(vf1);
}

void test_builtin_elementwise_exp(float f1, float f2, double d1, double d2,
                                  float4 vf1, float4 vf2) {
  // CHECK-LABEL: define void @test_builtin_elementwise_exp(
  // CHECK:      [[F1:%.+]] = load float, ptr %f1.addr, align 4
  // CHECK-NEXT:  call float @llvm.exp.f32(float [[F1]])
  f2 = __builtin_elementwise_exp(f1);

  // CHECK:      [[D1:%.+]] = load double, ptr %d1.addr, align 8
  // CHECK-NEXT: call double @llvm.exp.f64(double [[D1]])
  d2 = __builtin_elementwise_exp(d1);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, ptr %vf1.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.exp.v4f32(<4 x float> [[VF1]])
  vf2 = __builtin_elementwise_exp(vf1);
}

void test_builtin_elementwise_exp2(float f1, float f2, double d1, double d2,
                                  float4 vf1, float4 vf2) {
  // CHECK-LABEL: define void @test_builtin_elementwise_exp2(
  // CHECK:      [[F1:%.+]] = load float, ptr %f1.addr, align 4
  // CHECK-NEXT:  call float @llvm.exp2.f32(float [[F1]])
  f2 = __builtin_elementwise_exp2(f1);

  // CHECK:      [[D1:%.+]] = load double, ptr %d1.addr, align 8
  // CHECK-NEXT: call double @llvm.exp2.f64(double [[D1]])
  d2 = __builtin_elementwise_exp2(d1);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, ptr %vf1.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.exp2.v4f32(<4 x float> [[VF1]])
  vf2 = __builtin_elementwise_exp2(vf1);
}


void test_builtin_elementwise_floor(float f1, float f2, double d1, double d2,
                                    float4 vf1, float4 vf2) {
  // CHECK-LABEL: define void @test_builtin_elementwise_floor(
  // CHECK:      [[F1:%.+]] = load float, ptr %f1.addr, align 4
  // CHECK-NEXT:  call float @llvm.floor.f32(float [[F1]])
  f2 = __builtin_elementwise_floor(f1);

  // CHECK:      [[D1:%.+]] = load double, ptr %d1.addr, align 8
  // CHECK-NEXT: call double @llvm.floor.f64(double [[D1]])
  d2 = __builtin_elementwise_floor(d1);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, ptr %vf1.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.floor.v4f32(<4 x float> [[VF1]])
  vf2 = __builtin_elementwise_floor(vf1);
}

void test_builtin_elementwise_log(float f1, float f2, double d1, double d2,
                                  float4 vf1, float4 vf2) {
  // CHECK-LABEL: define void @test_builtin_elementwise_log(
  // CHECK:      [[F1:%.+]] = load float, ptr %f1.addr, align 4
  // CHECK-NEXT:  call float @llvm.log.f32(float [[F1]])
  f2 = __builtin_elementwise_log(f1);

  // CHECK:      [[D1:%.+]] = load double, ptr %d1.addr, align 8
  // CHECK-NEXT: call double @llvm.log.f64(double [[D1]])
  d2 = __builtin_elementwise_log(d1);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, ptr %vf1.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.log.v4f32(<4 x float> [[VF1]])
  vf2 = __builtin_elementwise_log(vf1);
}

void test_builtin_elementwise_log10(float f1, float f2, double d1, double d2,
                                  float4 vf1, float4 vf2) {
  // CHECK-LABEL: define void @test_builtin_elementwise_log10(
  // CHECK:      [[F1:%.+]] = load float, ptr %f1.addr, align 4
  // CHECK-NEXT:  call float @llvm.log10.f32(float [[F1]])
  f2 = __builtin_elementwise_log10(f1);

  // CHECK:      [[D1:%.+]] = load double, ptr %d1.addr, align 8
  // CHECK-NEXT: call double @llvm.log10.f64(double [[D1]])
  d2 = __builtin_elementwise_log10(d1);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, ptr %vf1.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.log10.v4f32(<4 x float> [[VF1]])
  vf2 = __builtin_elementwise_log10(vf1);
}

void test_builtin_elementwise_log2(float f1, float f2, double d1, double d2,
                                  float4 vf1, float4 vf2) {
  // CHECK-LABEL: define void @test_builtin_elementwise_log2(
  // CHECK:      [[F1:%.+]] = load float, ptr %f1.addr, align 4
  // CHECK-NEXT:  call float @llvm.log2.f32(float [[F1]])
  f2 = __builtin_elementwise_log2(f1);

  // CHECK:      [[D1:%.+]] = load double, ptr %d1.addr, align 8
  // CHECK-NEXT: call double @llvm.log2.f64(double [[D1]])
  d2 = __builtin_elementwise_log2(d1);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, ptr %vf1.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.log2.v4f32(<4 x float> [[VF1]])
  vf2 = __builtin_elementwise_log2(vf1);
}

void test_builtin_elementwise_popcount(si8 vi1, si8 vi2,
                                  long long int i1, long long int i2, short si,
                                  _BitInt(31) bi1, _BitInt(31) bi2) {

  
  // CHECK:      [[I1:%.+]] = load i64, ptr %i1.addr, align 8
  // CHECK-NEXT: call i64 @llvm.ctpop.i64(i64 [[I1]])
  i2 = __builtin_elementwise_popcount(i1);

  // CHECK:      [[VI1:%.+]] = load <8 x i16>, ptr %vi1.addr, align 16
  // CHECK-NEXT: call <8 x i16> @llvm.ctpop.v8i16(<8 x i16> [[VI1]])
  vi2 = __builtin_elementwise_popcount(vi1);

  // CHECK:      [[CVI2:%.+]] = load <8 x i16>, ptr %cvi2, align 16
  // CHECK-NEXT: call <8 x i16> @llvm.ctpop.v8i16(<8 x i16> [[CVI2]])
  const si8 cvi2 = vi2;
  vi2 = __builtin_elementwise_popcount(cvi2);

  // CHECK:      [[BI1:%.+]] = load i32, ptr %bi1.addr, align 4
  // CHECK-NEXT: [[LOADEDV:%.+]] = trunc i32 [[BI1]] to i31
  // CHECK-NEXT: call i31 @llvm.ctpop.i31(i31 [[LOADEDV]])
  bi2 = __builtin_elementwise_popcount(bi1);

  // CHECK:      [[IA1:%.+]] = load i32, ptr addrspace(1) @int_as_one, align 4
  // CHECK-NEXT: call i32 @llvm.ctpop.i32(i32 [[IA1]])
  b = __builtin_elementwise_popcount(int_as_one);

  // CHECK:   call i32 @llvm.ctpop.i32(i32 -10)
  b = __builtin_elementwise_popcount(-10);

  // CHECK:      [[SI:%.+]] = load i16, ptr %si.addr, align 2
  // CHECK-NEXT: [[SI_EXT:%.+]] = sext i16 [[SI]] to i32
  // CHECK-NEXT: [[RES:%.+]] = call i32 @llvm.ctpop.i32(i32 [[SI_EXT]])
  // CHECK-NEXT: = trunc i32 [[RES]] to i16
  si = __builtin_elementwise_popcount(si);
}

void test_builtin_elementwise_pow(float f1, float f2, double d1, double d2,
                                      float4 vf1, float4 vf2) {

  // CHECK-LABEL: define void @test_builtin_elementwise_pow(
  // CHECK:      [[F1:%.+]] = load float, ptr %f1.addr, align 4
  // CHECK:      [[F2:%.+]] = load float, ptr %f2.addr, align 4
  // CHECK-NEXT:  call float @llvm.pow.f32(float [[F1]], float [[F2]])
  f2 = __builtin_elementwise_pow(f1, f2);

  // CHECK:      [[D1:%.+]] = load double, ptr %d1.addr, align 8
  // CHECK:      [[D2:%.+]] = load double, ptr %d2.addr, align 8
  // CHECK-NEXT: call double @llvm.pow.f64(double [[D1]], double [[D2]])
  d2 = __builtin_elementwise_pow(d1, d2);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, ptr %vf1.addr, align 16
  // CHECK:      [[VF2:%.+]] = load <4 x float>, ptr %vf2.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.pow.v4f32(<4 x float> [[VF1]], <4 x float> [[VF2]])
  vf2 = __builtin_elementwise_pow(vf1, vf2);
}

void test_builtin_elementwise_roundeven(float f1, float f2, double d1, double d2,
                                        float4 vf1, float4 vf2) {
  // CHECK-LABEL: define void @test_builtin_elementwise_roundeven(
  // CHECK:      [[F1:%.+]] = load float, ptr %f1.addr, align 4
  // CHECK-NEXT:  call float @llvm.roundeven.f32(float [[F1]])
  f2 = __builtin_elementwise_roundeven(f1);

  // CHECK:      [[D1:%.+]] = load double, ptr %d1.addr, align 8
  // CHECK-NEXT: call double @llvm.roundeven.f64(double [[D1]])
  d2 = __builtin_elementwise_roundeven(d1);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, ptr %vf1.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.roundeven.v4f32(<4 x float> [[VF1]])
  vf2 = __builtin_elementwise_roundeven(vf1);
}

void test_builtin_elementwise_round(float f1, float f2, double d1, double d2,
                                        float4 vf1, float4 vf2) {
  // CHECK-LABEL: define void @test_builtin_elementwise_round(
  // CHECK:      [[F1:%.+]] = load float, ptr %f1.addr, align 4
  // CHECK-NEXT:  call float @llvm.round.f32(float [[F1]])
  f2 = __builtin_elementwise_round(f1);

  // CHECK:      [[D1:%.+]] = load double, ptr %d1.addr, align 8
  // CHECK-NEXT: call double @llvm.round.f64(double [[D1]])
  d2 = __builtin_elementwise_round(d1);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, ptr %vf1.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.round.v4f32(<4 x float> [[VF1]])
  vf2 = __builtin_elementwise_round(vf1);
}

void test_builtin_elementwise_rint(float f1, float f2, double d1, double d2,
                                   float4 vf1, float4 vf2) {
  // CHECK-LABEL: define void @test_builtin_elementwise_rint(
  // CHECK:      [[F1:%.+]] = load float, ptr %f1.addr, align 4
  // CHECK-NEXT:  call float @llvm.rint.f32(float [[F1]])
  f2 = __builtin_elementwise_rint(f1);

  // CHECK:      [[D1:%.+]] = load double, ptr %d1.addr, align 8
  // CHECK-NEXT: call double @llvm.rint.f64(double [[D1]])
  d2 = __builtin_elementwise_rint(d1);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, ptr %vf1.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.rint.v4f32(<4 x float> [[VF1]])
  vf2 = __builtin_elementwise_rint(vf1);
}

void test_builtin_elementwise_nearbyint(float f1, float f2, double d1, double d2,
                                        float4 vf1, float4 vf2) {
  // CHECK-LABEL: define void @test_builtin_elementwise_nearbyint(
  // CHECK:      [[F1:%.+]] = load float, ptr %f1.addr, align 4
  // CHECK-NEXT:  call float @llvm.nearbyint.f32(float [[F1]])
  f2 = __builtin_elementwise_nearbyint(f1);

  // CHECK:      [[D1:%.+]] = load double, ptr %d1.addr, align 8
  // CHECK-NEXT: call double @llvm.nearbyint.f64(double [[D1]])
  d2 = __builtin_elementwise_nearbyint(d1);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, ptr %vf1.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.nearbyint.v4f32(<4 x float> [[VF1]])
  vf2 = __builtin_elementwise_nearbyint(vf1);
}

void test_builtin_elementwise_sin(float f1, float f2, double d1, double d2,
                                  float4 vf1, float4 vf2) {
  // CHECK-LABEL: define void @test_builtin_elementwise_sin(
  // CHECK:      [[F1:%.+]] = load float, ptr %f1.addr, align 4
  // CHECK-NEXT:  call float @llvm.sin.f32(float [[F1]])
  f2 = __builtin_elementwise_sin(f1);

  // CHECK:      [[D1:%.+]] = load double, ptr %d1.addr, align 8
  // CHECK-NEXT: call double @llvm.sin.f64(double [[D1]])
  d2 = __builtin_elementwise_sin(d1);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, ptr %vf1.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.sin.v4f32(<4 x float> [[VF1]])
  vf2 = __builtin_elementwise_sin(vf1);
}

void test_builtin_elementwise_sinh(float f1, float f2, double d1, double d2,
                                  float4 vf1, float4 vf2) {
  // CHECK-LABEL: define void @test_builtin_elementwise_sinh(
  // CHECK:      [[F1:%.+]] = load float, ptr %f1.addr, align 4
  // CHECK-NEXT:  call float @llvm.sinh.f32(float [[F1]])
  f2 = __builtin_elementwise_sinh(f1);

  // CHECK:      [[D1:%.+]] = load double, ptr %d1.addr, align 8
  // CHECK-NEXT: call double @llvm.sinh.f64(double [[D1]])
  d2 = __builtin_elementwise_sinh(d1);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, ptr %vf1.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.sinh.v4f32(<4 x float> [[VF1]])
  vf2 = __builtin_elementwise_sinh(vf1);
}

void test_builtin_elementwise_sqrt(float f1, float f2, double d1, double d2,
                                  float4 vf1, float4 vf2) {
  // CHECK-LABEL: define void @test_builtin_elementwise_sqrt(
  // CHECK:      [[F1:%.+]] = load float, ptr %f1.addr, align 4
  // CHECK-NEXT:  call float @llvm.sqrt.f32(float [[F1]])
  f2 = __builtin_elementwise_sqrt(f1);

  // CHECK:      [[D1:%.+]] = load double, ptr %d1.addr, align 8
  // CHECK-NEXT: call double @llvm.sqrt.f64(double [[D1]])
  d2 = __builtin_elementwise_sqrt(d1);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, ptr %vf1.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.sqrt.v4f32(<4 x float> [[VF1]])
  vf2 = __builtin_elementwise_sqrt(vf1);
}

void test_builtin_elementwise_tan(float f1, float f2, double d1, double d2,
                                  float4 vf1, float4 vf2) {
  // CHECK-LABEL: define void @test_builtin_elementwise_tan(
  // CHECK:      [[F1:%.+]] = load float, ptr %f1.addr, align 4
  // CHECK-NEXT:  call float @llvm.tan.f32(float [[F1]])
  f2 = __builtin_elementwise_tan(f1);

  // CHECK:      [[D1:%.+]] = load double, ptr %d1.addr, align 8
  // CHECK-NEXT: call double @llvm.tan.f64(double [[D1]])
  d2 = __builtin_elementwise_tan(d1);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, ptr %vf1.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.tan.v4f32(<4 x float> [[VF1]])
  vf2 = __builtin_elementwise_tan(vf1);
}

void test_builtin_elementwise_tanh(float f1, float f2, double d1, double d2,
                                  float4 vf1, float4 vf2) {
  // CHECK-LABEL: define void @test_builtin_elementwise_tanh(
  // CHECK:      [[F1:%.+]] = load float, ptr %f1.addr, align 4
  // CHECK-NEXT:  call float @llvm.tanh.f32(float [[F1]])
  f2 = __builtin_elementwise_tanh(f1);

  // CHECK:      [[D1:%.+]] = load double, ptr %d1.addr, align 8
  // CHECK-NEXT: call double @llvm.tanh.f64(double [[D1]])
  d2 = __builtin_elementwise_tanh(d1);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, ptr %vf1.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.tanh.v4f32(<4 x float> [[VF1]])
  vf2 = __builtin_elementwise_tanh(vf1);
}

void test_builtin_elementwise_trunc(float f1, float f2, double d1, double d2,
                                    float4 vf1, float4 vf2) {
  // CHECK-LABEL: define void @test_builtin_elementwise_trunc(
  // CHECK:      [[F1:%.+]] = load float, ptr %f1.addr, align 4
  // CHECK-NEXT:  call float @llvm.trunc.f32(float [[F1]])
  f2 = __builtin_elementwise_trunc(f1);

  // CHECK:      [[D1:%.+]] = load double, ptr %d1.addr, align 8
  // CHECK-NEXT: call double @llvm.trunc.f64(double [[D1]])
  d2 = __builtin_elementwise_trunc(d1);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, ptr %vf1.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.trunc.v4f32(<4 x float> [[VF1]])
  vf2 = __builtin_elementwise_trunc(vf1);
}

void test_builtin_elementwise_canonicalize(float f1, float f2, double d1, double d2,
                                           float4 vf1, float4 vf2) {
  // CHECK-LABEL: define void @test_builtin_elementwise_canonicalize(
  // CHECK:      [[F1:%.+]] = load float, ptr %f1.addr, align 4
  // CHECK-NEXT:  call float @llvm.canonicalize.f32(float [[F1]])
  f2 = __builtin_elementwise_canonicalize(f1);

  // CHECK:      [[D1:%.+]] = load double, ptr %d1.addr, align 8
  // CHECK-NEXT: call double @llvm.canonicalize.f64(double [[D1]])
  d2 = __builtin_elementwise_canonicalize(d1);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, ptr %vf1.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.canonicalize.v4f32(<4 x float> [[VF1]])
  vf2 = __builtin_elementwise_canonicalize(vf1);
}

void test_builtin_elementwise_copysign(float f1, float f2, double d1, double d2,
                                       float4 vf1, float4 vf2, double2 v2f64) {
  // CHECK-LABEL: define void @test_builtin_elementwise_copysign(
  // CHECK:      [[F1:%.+]] = load float, ptr %f1.addr, align 4
  // CHECK-NEXT: [[F2:%.+]] = load float, ptr %f2.addr, align 4
  // CHECK-NEXT:  call float @llvm.copysign.f32(float %0, float %1)
  f1 = __builtin_elementwise_copysign(f1, f2);

  // CHECK:      [[D1:%.+]] = load double, ptr %d1.addr, align 8
  // CHECK-NEXT: [[D2:%.+]] = load double, ptr %d2.addr, align 8
  // CHECK-NEXT: call double @llvm.copysign.f64(double [[D1]], double [[D2]])
  d1 = __builtin_elementwise_copysign(d1, d2);

  // CHECK:      [[D1:%.+]] = load double, ptr %d1.addr, align 8
  // CHECK-NEXT: call double @llvm.copysign.f64(double [[D1]], double 2.000000e+00)
  d1 = __builtin_elementwise_copysign(d1, 2.0);

  // CHECK:      [[VF1:%.+]] = load <4 x float>, ptr %vf1.addr, align 16
  // CHECK-NEXT: [[VF2:%.+]] = load <4 x float>, ptr %vf2.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.copysign.v4f32(<4 x float> [[VF1]], <4 x float> [[VF2]])
  vf1 = __builtin_elementwise_copysign(vf1, vf2);

  // CHECK:      [[CVF1:%.+]] = load <4 x float>, ptr %cvf1, align 16
  // CHECK-NEXT: [[VF2:%.+]] = load <4 x float>, ptr %vf2.addr, align 16
  // CHECK-NEXT: call <4 x float> @llvm.copysign.v4f32(<4 x float> [[CVF1]], <4 x float> [[VF2]])
  const float4 cvf1 = vf1;
  vf1 = __builtin_elementwise_copysign(cvf1, vf2);

  // CHECK:      [[VF2:%.+]] = load <4 x float>, ptr %vf2.addr, align 16
  // CHECK-NEXT: [[CVF1:%.+]] = load <4 x float>, ptr %cvf1, align 16
  // CHECK-NEXT: call <4 x float> @llvm.copysign.v4f32(<4 x float> [[VF2]], <4 x float> [[CVF1]])
  vf1 = __builtin_elementwise_copysign(vf2, cvf1);


  // CHECK:      [[F1:%.+]] = load float, ptr %f1.addr
  // CHECK-NEXT: call float @llvm.copysign.f32(float [[F1]], float 2.000000e+00)
  f1 = __builtin_elementwise_copysign(f1, 2.0f);

  // CHECK:      [[F1:%.+]] = load float, ptr %f1.addr
  // CHECK-NEXT: call float @llvm.copysign.f32(float 2.000000e+00, float [[F1]])
  f1 = __builtin_elementwise_copysign(2.0f, f1);

  // CHECK:      [[V2F64:%.+]] = load <2 x double>, ptr %v2f64.addr, align 16
  // CHECK-NEXT: call <2 x double> @llvm.copysign.v2f64(<2 x double> <double 1.000000e+00, double 1.000000e+00>, <2 x double> [[V2F64]])
  v2f64 = __builtin_elementwise_copysign((double2)1.0, v2f64);
}

void test_builtin_elementwise_fma(float f32, double f64,
                                  float2 v2f32, float4 v4f32,
                                  double2 v2f64, double3 v3f64,
                                  const float4 c_v4f32,
                                  half f16, half2 v2f16) {
  // CHECK-LABEL: define void @test_builtin_elementwise_fma(
  // CHECK:      [[F32_0:%.+]] = load float, ptr %f32.addr
  // CHECK-NEXT: [[F32_1:%.+]] = load float, ptr %f32.addr
  // CHECK-NEXT: [[F32_2:%.+]] = load float, ptr %f32.addr
  // CHECK-NEXT: call float @llvm.fma.f32(float [[F32_0]], float [[F32_1]], float [[F32_2]])
  float f2 = __builtin_elementwise_fma(f32, f32, f32);

  // CHECK:      [[F64_0:%.+]] = load double, ptr %f64.addr
  // CHECK-NEXT: [[F64_1:%.+]] = load double, ptr %f64.addr
  // CHECK-NEXT: [[F64_2:%.+]] = load double, ptr %f64.addr
  // CHECK-NEXT: call double @llvm.fma.f64(double [[F64_0]], double [[F64_1]], double [[F64_2]])
  double d2 = __builtin_elementwise_fma(f64, f64, f64);

  // CHECK:      [[V4F32_0:%.+]] = load <4 x float>, ptr %v4f32.addr
  // CHECK-NEXT: [[V4F32_1:%.+]] = load <4 x float>, ptr %v4f32.addr
  // CHECK-NEXT: [[V4F32_2:%.+]] = load <4 x float>, ptr %v4f32.addr
  // CHECK-NEXT: call <4 x float> @llvm.fma.v4f32(<4 x float> [[V4F32_0]], <4 x float> [[V4F32_1]], <4 x float> [[V4F32_2]])
  float4 tmp_v4f32 = __builtin_elementwise_fma(v4f32, v4f32, v4f32);


  // FIXME: Are we really still doing the 3 vector load workaround
  // CHECK:      [[V3F64_LOAD_0:%.+]] = load <4 x double>, ptr %v3f64.addr
  // CHECK-NEXT: [[V3F64_0:%.+]] = shufflevector
  // CHECK-NEXT: [[V3F64_LOAD_1:%.+]] = load <4 x double>, ptr %v3f64.addr
  // CHECK-NEXT: [[V3F64_1:%.+]] = shufflevector
  // CHECK-NEXT: [[V3F64_LOAD_2:%.+]] = load <4 x double>, ptr %v3f64.addr
  // CHECK-NEXT: [[V3F64_2:%.+]] = shufflevector
    // CHECK-NEXT: call <3 x double> @llvm.fma.v3f64(<3 x double> [[V3F64_0]], <3 x double> [[V3F64_1]], <3 x double> [[V3F64_2]])
  v3f64 = __builtin_elementwise_fma(v3f64, v3f64, v3f64);

  // CHECK:      [[F64_0:%.+]] = load double, ptr %f64.addr
  // CHECK-NEXT: [[F64_1:%.+]] = load double, ptr %f64.addr
  // CHECK-NEXT: [[F64_2:%.+]] = load double, ptr %f64.addr
  // CHECK-NEXT: call double @llvm.fma.f64(double [[F64_0]], double [[F64_1]], double [[F64_2]])
  v2f64 = __builtin_elementwise_fma(f64, f64, f64);

  // CHECK:      [[V4F32_0:%.+]] = load <4 x float>, ptr %c_v4f32.addr
  // CHECK-NEXT: [[V4F32_1:%.+]] = load <4 x float>, ptr %c_v4f32.addr
  // CHECK-NEXT: [[V4F32_2:%.+]] = load <4 x float>, ptr %c_v4f32.addr
  // CHECK-NEXT: call <4 x float> @llvm.fma.v4f32(<4 x float> [[V4F32_0]], <4 x float> [[V4F32_1]], <4 x float> [[V4F32_2]])
  v4f32 = __builtin_elementwise_fma(c_v4f32, c_v4f32, c_v4f32);

  // CHECK:      [[F16_0:%.+]] = load half, ptr %f16.addr
  // CHECK-NEXT: [[F16_1:%.+]] = load half, ptr %f16.addr
  // CHECK-NEXT: [[F16_2:%.+]] = load half, ptr %f16.addr
  // CHECK-NEXT: call half @llvm.fma.f16(half [[F16_0]], half [[F16_1]], half [[F16_2]])
  half tmp_f16 = __builtin_elementwise_fma(f16, f16, f16);

  // CHECK:      [[V2F16_0:%.+]] = load <2 x half>, ptr %v2f16.addr
  // CHECK-NEXT: [[V2F16_1:%.+]] = load <2 x half>, ptr %v2f16.addr
  // CHECK-NEXT: [[V2F16_2:%.+]] = load <2 x half>, ptr %v2f16.addr
  // CHECK-NEXT: call <2 x half> @llvm.fma.v2f16(<2 x half> [[V2F16_0]], <2 x half> [[V2F16_1]], <2 x half> [[V2F16_2]])
  half2 tmp0_v2f16 = __builtin_elementwise_fma(v2f16, v2f16, v2f16);

  // CHECK:      [[V2F16_0:%.+]] = load <2 x half>, ptr %v2f16.addr
  // CHECK-NEXT: [[V2F16_1:%.+]] = load <2 x half>, ptr %v2f16.addr
  // CHECK-NEXT: [[F16_2:%.+]] = load half, ptr %f16.addr
  // CHECK-NEXT: [[V2F16_2_INSERT:%.+]] = insertelement
  // CHECK-NEXT: [[V2F16_2:%.+]] = shufflevector <2 x half> [[V2F16_2_INSERT]], <2 x half> poison, <2 x i32> zeroinitializer
  // CHECK-NEXT: call <2 x half> @llvm.fma.v2f16(<2 x half> [[V2F16_0]], <2 x half> [[V2F16_1]], <2 x half> [[V2F16_2]])
  half2 tmp1_v2f16 = __builtin_elementwise_fma(v2f16, v2f16, (half2)f16);

  // CHECK:      [[V2F16_0:%.+]] = load <2 x half>, ptr %v2f16.addr
  // CHECK-NEXT: [[V2F16_1:%.+]] = load <2 x half>, ptr %v2f16.addr
  // CHECK-NEXT: call <2 x half> @llvm.fma.v2f16(<2 x half> [[V2F16_0]], <2 x half> [[V2F16_1]], <2 x half> <half 0xH4400, half 0xH4400>)
  half2 tmp2_v2f16 = __builtin_elementwise_fma(v2f16, v2f16, (half2)4.0);

}
