// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type -fnative-int16-type -emit-llvm \
// RUN:   -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,NATIVE_HALF

// CHECK-LABEL: test_select_float1x2
// CHECK: [[SELECT:%.*]] = select {{.*}}<2 x i1> {{%.*}}, <2 x float> {{%.*}}, <2 x float> {{%.*}}
// CHECK: ret <2 x float> [[SELECT]]
float1x2 test_select_float1x2(bool1x2 cond, float1x2 tVals, float1x2 fVals) {
  return select(cond, tVals, fVals);
}

// CHECK-LABEL: test_select_float1x3
// CHECK: [[SELECT:%.*]] = select {{.*}}<3 x i1> {{%.*}}, <3 x float> {{%.*}}, <3 x float> {{%.*}}
// CHECK: ret <3 x float> [[SELECT]]
float1x3 test_select_float1x3(bool1x3 cond, float1x3 tVals, float1x3 fVals) {
  return select(cond, tVals, fVals);
}

// CHECK-LABEL: test_select_float1x4
// CHECK: [[SELECT:%.*]] = select {{.*}}<4 x i1> {{%.*}}, <4 x float> {{%.*}}, <4 x float> {{%.*}}
// CHECK: ret <4 x float> [[SELECT]]
float1x4 test_select_float1x4(bool1x4 cond, float1x4 tVals, float1x4 fVals) {
  return select(cond, tVals, fVals);
}

// CHECK-LABEL: test_select_float2x1
// CHECK: [[SELECT:%.*]] = select {{.*}}<2 x i1> {{%.*}}, <2 x float> {{%.*}}, <2 x float> {{%.*}}
// CHECK: ret <2 x float> [[SELECT]]
float2x1 test_select_float2x1(bool2x1 cond, float2x1 tVals, float2x1 fVals) {
  return select(cond, tVals, fVals);
}

// CHECK-LABEL: test_select_float2x2
// CHECK: [[SELECT:%.*]] = select {{.*}}<4 x i1> {{%.*}}, <4 x float> {{%.*}}, <4 x float> {{%.*}}
// CHECK: ret <4 x float> [[SELECT]]
float2x2 test_select_float2x2(bool2x2 cond, float2x2 tVals, float2x2 fVals) {
  return select(cond, tVals, fVals);
}

// CHECK-LABEL: test_select_float2x3
// CHECK: [[SELECT:%.*]] = select {{.*}}<6 x i1> {{%.*}}, <6 x float> {{%.*}}, <6 x float> {{%.*}}
// CHECK: ret <6 x float> [[SELECT]]
float2x3 test_select_float2x3(bool2x3 cond, float2x3 tVals, float2x3 fVals) {
  return select(cond, tVals, fVals);
}

// CHECK-LABEL: test_select_float2x4
// CHECK: [[SELECT:%.*]] = select {{.*}}<8 x i1> {{%.*}}, <8 x float> {{%.*}}, <8 x float> {{%.*}}
// CHECK: ret <8 x float> [[SELECT]]
float2x4 test_select_float2x4(bool2x4 cond, float2x4 tVals, float2x4 fVals) {
  return select(cond, tVals, fVals);
}

// CHECK-LABEL: test_select_float3x1
// CHECK: [[SELECT:%.*]] = select {{.*}}<3 x i1> {{%.*}}, <3 x float> {{%.*}}, <3 x float> {{%.*}}
// CHECK: ret <3 x float> [[SELECT]]
float3x1 test_select_float3x1(bool3x1 cond, float3x1 tVals, float3x1 fVals) {
  return select(cond, tVals, fVals);
}

// CHECK-LABEL: test_select_float3x2
// CHECK: [[SELECT:%.*]] = select {{.*}}<6 x i1> {{%.*}}, <6 x float> {{%.*}}, <6 x float> {{%.*}}
// CHECK: ret <6 x float> [[SELECT]]
float3x2 test_select_float3x2(bool3x2 cond, float3x2 tVals, float3x2 fVals) {
  return select(cond, tVals, fVals);
}

// CHECK-LABEL: test_select_float3x3
// CHECK: [[SELECT:%.*]] = select {{.*}}<9 x i1> {{%.*}}, <9 x float> {{%.*}}, <9 x float> {{%.*}}
// CHECK: ret <9 x float> [[SELECT]]
float3x3 test_select_float3x3(bool3x3 cond, float3x3 tVals, float3x3 fVals) {
  return select(cond, tVals, fVals);
}

// CHECK-LABEL: test_select_float3x4
// CHECK: [[SELECT:%.*]] = select {{.*}}<12 x i1> {{%.*}}, <12 x float> {{%.*}}, <12 x float> {{%.*}}
// CHECK: ret <12 x float> [[SELECT]]
float3x4 test_select_float3x4(bool3x4 cond, float3x4 tVals, float3x4 fVals) {
  return select(cond, tVals, fVals);
}

// CHECK-LABEL: test_select_float4x1
// CHECK: [[SELECT:%.*]] = select {{.*}}<4 x i1> {{%.*}}, <4 x float> {{%.*}}, <4 x float> {{%.*}}
// CHECK: ret <4 x float> [[SELECT]]
float4x1 test_select_float4x1(bool4x1 cond, float4x1 tVals, float4x1 fVals) {
  return select(cond, tVals, fVals);
}

// CHECK-LABEL: test_select_float4x2
// CHECK: [[SELECT:%.*]] = select {{.*}}<8 x i1> {{%.*}}, <8 x float> {{%.*}}, <8 x float> {{%.*}}
// CHECK: ret <8 x float> [[SELECT]]
float4x2 test_select_float4x2(bool4x2 cond, float4x2 tVals, float4x2 fVals) {
  return select(cond, tVals, fVals);
}

// CHECK-LABEL: test_select_float4x3
// CHECK: [[SELECT:%.*]] = select {{.*}}<12 x i1> {{%.*}}, <12 x float> {{%.*}}, <12 x float> {{%.*}}
// CHECK: ret <12 x float> [[SELECT]]
float4x3 test_select_float4x3(bool4x3 cond, float4x3 tVals, float4x3 fVals) {
  return select(cond, tVals, fVals);
}

// CHECK-LABEL: test_select_float4x4
// CHECK: [[SELECT:%.*]] = select {{.*}}<16 x i1> {{%.*}}, <16 x float> {{%.*}}, <16 x float> {{%.*}}
// CHECK: ret <16 x float> [[SELECT]]
float4x4 test_select_float4x4(bool4x4 cond, float4x4 tVals, float4x4 fVals) {
  return select(cond, tVals, fVals);
}

#ifdef __HLSL_ENABLE_16_BIT
// NATIVE_HALF-LABEL: test_select_short2x2
// NATIVE_HALF: [[SELECT:%.*]] = select {{.*}}<4 x i1> {{%.*}}, <4 x i16> {{%.*}}, <4 x i16> {{%.*}}
// NATIVE_HALF: ret <4 x i16> [[SELECT]]
int16_t2x2 test_select_short2x2(bool2x2 cond, int16_t2x2 tVals, int16_t2x2 fVals) {
  return select(cond, tVals, fVals);
}

// NATIVE_HALF-LABEL: test_select_ushort2x2
// NATIVE_HALF: [[SELECT:%.*]] = select {{.*}}<4 x i1> {{%.*}}, <4 x i16> {{%.*}}, <4 x i16> {{%.*}}
// NATIVE_HALF: ret <4 x i16> [[SELECT]]
uint16_t2x2 test_select_ushort2x2(bool2x2 cond, uint16_t2x2 tVals, uint16_t2x2 fVals) {
  return select(cond, tVals, fVals);
}
#endif

// CHECK-LABEL: test_select_int2x2
// CHECK: [[SELECT:%.*]] = select {{.*}}<4 x i1> {{%.*}}, <4 x i32> {{%.*}}, <4 x i32> {{%.*}}
// CHECK: ret <4 x i32> [[SELECT]]
int2x2 test_select_int2x2(bool2x2 cond, int2x2 tVals, int2x2 fVals) {
  return select(cond, tVals, fVals);
}

// CHECK-LABEL: test_select_uint2x2
// CHECK: [[SELECT:%.*]] = select {{.*}}<4 x i1> {{%.*}}, <4 x i32> {{%.*}}, <4 x i32> {{%.*}}
// CHECK: ret <4 x i32> [[SELECT]]
uint2x2 test_select_uint2x2(bool2x2 cond, uint2x2 tVals, uint2x2 fVals) {
  return select(cond, tVals, fVals);
}

// CHECK-LABEL: test_select_long2x2
// CHECK: [[SELECT:%.*]] = select {{.*}}<4 x i1> {{%.*}}, <4 x i64> {{%.*}}, <4 x i64> {{%.*}}
// CHECK: ret <4 x i64> [[SELECT]]
int64_t2x2 test_select_long2x2(bool2x2 cond, int64_t2x2 tVals, int64_t2x2 fVals) {
  return select(cond, tVals, fVals);
}

// CHECK-LABEL: test_select_ulong2x2
// CHECK: [[SELECT:%.*]] = select {{.*}}<4 x i1> {{%.*}}, <4 x i64> {{%.*}}, <4 x i64> {{%.*}}
// CHECK: ret <4 x i64> [[SELECT]]
uint64_t2x2 test_select_ulong2x2(bool2x2 cond, uint64_t2x2 tVals, uint64_t2x2 fVals) {
  return select(cond, tVals, fVals);
}

// CHECK-LABEL: test_select_half2x2
// NO_HALF: [[SELECT:%.*]] = select {{.*}}<4 x i1> {{%.*}}, <4 x float> {{%.*}}, <4 x float> {{%.*}}
// NO_HALF: ret <4 x float> [[SELECT]]
// NATIVE_HALF: [[SELECT:%.*]] = select {{.*}}<4 x i1> {{%.*}}, <4 x half> {{%.*}}, <4 x half> {{%.*}}
// NATIVE_HALF: ret <4 x half> [[SELECT]]
half2x2 test_select_half2x2(bool2x2 cond, half2x2 tVals, half2x2 fVals) {
  return select(cond, tVals, fVals);
}

// CHECK-LABEL: test_select_double2x2
// CHECK: [[SELECT:%.*]] = select {{.*}}<4 x i1> {{%.*}}, <4 x double> {{%.*}}, <4 x double> {{%.*}}
// CHECK: ret <4 x double> [[SELECT]]
double2x2 test_select_double2x2(bool2x2 cond, double2x2 tVals, double2x2 fVals) {
  return select(cond, tVals, fVals);
}

// CHECK-LABEL: test_select_bool2x2
// CHECK-COUNT-3: icmp ne <4 x i32> {{%.*}}, zeroinitializer
// CHECK: [[SELECT:%.*]] = select <4 x i1> {{%.*}}, <4 x i1> {{%.*}}, <4 x i1> {{%.*}}
// CHECK: ret <4 x i1> [[SELECT]]
bool2x2 test_select_bool2x2(bool2x2 cond, bool2x2 tVals, bool2x2 fVals) {
  return select(cond, tVals, fVals);
}

// CHECK-LABEL: test_select_matrix_scalar_matrix
// CHECK: [[SPLAT_SRC:%.*]] = insertelement <4 x float> poison, float {{%.*}}, i64 0
// CHECK: [[SPLAT:%.*]] = shufflevector <4 x float> [[SPLAT_SRC]], <4 x float> poison, <4 x i32> zeroinitializer
// CHECK: [[SELECT:%.*]] = select {{.*}}<4 x i1> {{%.*}}, <4 x float> [[SPLAT]], <4 x float> {{%.*}}
// CHECK: ret <4 x float> [[SELECT]]
float2x2 test_select_matrix_scalar_matrix(bool2x2 cond, float tVal,
                                               float2x2 fVals) {
  return select(cond, tVal, fVals);
}

// CHECK-LABEL: test_select_matrix_matrix_scalar
// CHECK: [[SPLAT_SRC:%.*]] = insertelement <4 x float> poison, float {{%.*}}, i64 0
// CHECK: [[SPLAT:%.*]] = shufflevector <4 x float> [[SPLAT_SRC]], <4 x float> poison, <4 x i32> zeroinitializer
// CHECK: [[SELECT:%.*]] = select {{.*}}<4 x i1> {{%.*}}, <4 x float> {{%.*}}, <4 x float> [[SPLAT]]
// CHECK: ret <4 x float> [[SELECT]]
float2x2 test_select_matrix_matrix_scalar(bool2x2 cond, float2x2 tVals,
                                               float fVal) {
  return select(cond, tVals, fVal);
}

// CHECK-LABEL: test_select_matrix_scalar_scalar
// CHECK: [[SPLAT_SRC1:%.*]] = insertelement <4 x float> poison, float {{%.*}}, i64 0
// CHECK: [[SPLAT1:%.*]] = shufflevector <4 x float> [[SPLAT_SRC1]], <4 x float> poison, <4 x i32> zeroinitializer
// CHECK: [[SPLAT_SRC2:%.*]] = insertelement <4 x float> poison, float {{%.*}}, i64 0
// CHECK: [[SPLAT2:%.*]] = shufflevector <4 x float> [[SPLAT_SRC2]], <4 x float> poison, <4 x i32> zeroinitializer
// CHECK: [[SELECT:%.*]] = select {{.*}}<4 x i1> {{%.*}}, <4 x float> [[SPLAT1]], <4 x float> [[SPLAT2]]
// CHECK: ret <4 x float> [[SELECT]]
float2x2 test_select_matrix_scalar_scalar(bool2x2 cond, float tVal,
                                               float fVal) {
  return select(cond, tVal, fVal);
}

// CHECK-LABEL: test_select_int_condition
// CHECK: [[COND:%.*]] = load <4 x i32>, ptr {{%.*}}, align 4
// CHECK: [[TOBOOL:%.*]] = icmp ne <4 x i32> [[COND]], zeroinitializer
// CHECK: [[SELECT:%.*]] = select <4 x i1> [[TOBOOL]], <4 x i32> {{%.*}}, <4 x i32> {{%.*}}
// CHECK: ret <4 x i32> [[SELECT]]
int2x2 test_select_int_condition(int2x2 cond, int2x2 tVals, int2x2 fVals) {
  return select(cond, tVals, fVals);
}

// CHECK-LABEL: test_select_float_condition
// CHECK: [[COND:%.*]] = load <4 x float>, ptr {{%.*}}, align 4
// CHECK: [[TOBOOL:%.*]] = fcmp {{.*}}une <4 x float> [[COND]], zeroinitializer
// CHECK: [[SELECT:%.*]] = select {{.*}}<4 x i1> [[TOBOOL]], <4 x float> {{%.*}}, <4 x float> {{%.*}}
// CHECK: ret <4 x float> [[SELECT]]
float2x2 test_select_float_condition(float2x2 cond, float2x2 tVals,
                                          float2x2 fVals) {
  return select(cond, tVals, fVals);
}
