// RUN: %clang_cc1 %s -triple arm64-apple-ios8.1.0 -std=c++11 -emit-llvm -o - | FileCheck %s

typedef __attribute__((__ext_vector_type__(8))) float vector_float8;

typedef vector_float8 float8;

// rdar://20000762
// CHECK-LABEL: define void @_Z23MandelbrotPolyCalcSIMD8v
void MandelbrotPolyCalcSIMD8() {
  constexpr float8 v4 = 4.0;  // value to compare against abs(z)^2, to see if bounded
  float8 vABS;
  auto vLT  = vABS < v4;
  // CHECK: store <8 x float>
  // CHECK: [[ZERO:%.*]] = load <8 x float>, <8 x float>* [[VARBS:%.*]]
  // CHECK: [[CMP:%.*]] = fcmp olt <8 x float> [[ZERO]]
  // CHECK: [[SEXT:%.*]] = sext <8 x i1> [[CMP]] to <8 x i32>
  // CHECK: store <8 x i32> [[SEXT]], <8 x i32>* [[VLT:%.*]]
}

typedef __attribute__((__ext_vector_type__(4))) int int4;
typedef __attribute__((__ext_vector_type__(4))) float float4;
typedef __attribute__((__ext_vector_type__(4))) __int128 bigint4;

// CHECK-LABEL: define void @_Z14BoolConversionv
void BoolConversion() {
  // CHECK: store <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>
  int4 intsT = (int4)true;
  // CHECK: store <4 x i32> zeroinitializer
  int4 intsF = (int4)false;
  // CHECK: store <4 x float> <float -1.000000e+00, float -1.000000e+00, float -1.000000e+00, float -1.000000e+00>
  float4 floatsT = (float4)true;
  // CHECK: store <4 x float> zeroinitializer
  float4 floatsF = (float4)false;
  // CHECK: store <4 x i128> <i128 -1, i128 -1, i128 -1, i128 -1>
  bigint4 bigintsT = (bigint4)true;
  // CHECK: store <4 x i128> zeroinitializer
  bigint4 bigintsF = (bigint4)false;

  // CHECK: store <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>
  constexpr int4 cIntsT = (int4)true;
  // CHECK: store <4 x i32> zeroinitializer
  constexpr int4 cIntsF = (int4)false;
  // CHECK: store <4 x float> <float -1.000000e+00, float -1.000000e+00, float -1.000000e+00, float -1.000000e+00>
  constexpr float4 cFloatsT = (float4)true;
  // CHECK: store <4 x float> zeroinitializer
  constexpr float4 cFloatsF = (float4)false;
  // CHECK: store <4 x i128> <i128 -1, i128 -1, i128 -1, i128 -1>
  constexpr bigint4 cBigintsT = (bigint4)true;
  // CHECK: store <4 x i128> zeroinitializer
  constexpr bigint4 cBigintsF = (bigint4)false;
}
