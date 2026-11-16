// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type -emit-llvm \
// RUN:   -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,NATIVE_HALF

// CHECK-LABEL: test_select_half2x3
// NO_HALF: select {{.*}}<6 x i1> {{%.*}}, <6 x float> {{%.*}}, <6 x float> {{%.*}}
// NATIVE_HALF: select {{.*}}<6 x i1> {{%.*}}, <6 x half> {{%.*}}, <6 x half> {{%.*}}
half2x3 test_select_half2x3(bool2x3 cond, half2x3 tVals, half2x3 fVals) {
  return select(cond, tVals, fVals);
}

// CHECK-LABEL: test_select_half3x4
// NO_HALF: select {{.*}}<12 x i1> {{%.*}}, <12 x float> {{%.*}}, <12 x float> {{%.*}}
// NATIVE_HALF: select {{.*}}<12 x i1> {{%.*}}, <12 x half> {{%.*}}, <12 x half> {{%.*}}
half3x4 test_select_half3x4(bool3x4 cond, half3x4 tVals, half3x4 fVals) {
  return select(cond, tVals, fVals);
}

// CHECK-LABEL: test_select_half4x4
// NO_HALF: select {{.*}}<16 x i1> {{%.*}}, <16 x float> {{%.*}}, <16 x float> {{%.*}}
// NATIVE_HALF: select {{.*}}<16 x i1> {{%.*}}, <16 x half> {{%.*}}, <16 x half> {{%.*}}
half4x4 test_select_half4x4(bool4x4 cond, half4x4 tVals, half4x4 fVals) {
  return select(cond, tVals, fVals);
}

// CHECK-LABEL: test_select_bool2x2
// CHECK: [[COND:%.*]] = icmp ne <4 x i32> {{%.*}}, zeroinitializer
// CHECK: [[TVALS:%.*]] = icmp ne <4 x i32> {{%.*}}, zeroinitializer
// CHECK: [[FVALS:%.*]] = icmp ne <4 x i32> {{%.*}}, zeroinitializer
// CHECK: select <4 x i1> [[COND]], <4 x i1> [[TVALS]], <4 x i1> [[FVALS]]
bool2x2 test_select_bool2x2(bool2x2 cond, bool2x2 tVals, bool2x2 fVals) {
  return select(cond, tVals, fVals);
}

// CHECK-LABEL: test_select_matrix_scalar_matrix
// CHECK: [[TVAL:%.*]] = load float, ptr %tVal.addr
// CHECK: [[SPLAT_SRC:%.*]] = insertelement <4 x float> poison, float [[TVAL]], i64 0
// CHECK: [[SPLAT:%.*]] = shufflevector <4 x float> [[SPLAT_SRC]], <4 x float> poison, <4 x i32> zeroinitializer
// CHECK: select {{.*}}<4 x i1> {{%.*}}, <4 x float> [[SPLAT]], <4 x float> {{%.*}}
float2x2 test_select_matrix_scalar_matrix(bool2x2 cond, float tVal,
                                               float2x2 fVals) {
  return select(cond, tVal, fVals);
}

// CHECK-LABEL: test_select_matrix_matrix_scalar
// CHECK: [[FVAL:%.*]] = load float, ptr %fVal.addr
// CHECK: [[SPLAT_SRC:%.*]] = insertelement <4 x float> poison, float [[FVAL]], i64 0
// CHECK: [[SPLAT:%.*]] = shufflevector <4 x float> [[SPLAT_SRC]], <4 x float> poison, <4 x i32> zeroinitializer
// CHECK: select {{.*}}<4 x i1> {{%.*}}, <4 x float> {{%.*}}, <4 x float> [[SPLAT]]
float2x2 test_select_matrix_matrix_scalar(bool2x2 cond, float2x2 tVals,
                                               float fVal) {
  return select(cond, tVals, fVal);
}

// CHECK-LABEL: test_select_matrix_scalar_scalar
// CHECK: [[TVAL:%.*]] = load float, ptr %tVal.addr
// CHECK: [[FVAL:%.*]] = load float, ptr %fVal.addr
// CHECK: [[SPLAT_SRC1:%.*]] = insertelement <4 x float> poison, float [[TVAL]], i64 0
// CHECK: [[SPLAT1:%.*]] = shufflevector <4 x float> [[SPLAT_SRC1]], <4 x float> poison, <4 x i32> zeroinitializer
// CHECK: [[SPLAT_SRC2:%.*]] = insertelement <4 x float> poison, float [[FVAL]], i64 0
// CHECK: [[SPLAT2:%.*]] = shufflevector <4 x float> [[SPLAT_SRC2]], <4 x float> poison, <4 x i32> zeroinitializer
// CHECK: select {{.*}}<4 x i1> {{%.*}}, <4 x float> [[SPLAT1]], <4 x float> [[SPLAT2]]
float2x2 test_select_matrix_scalar_scalar(bool2x2 cond, float tVal,
                                               float fVal) {
  return select(cond, tVal, fVal);
}

// CHECK-LABEL: test_select_int_condition
// CHECK: [[COND:%.*]] = load <4 x i32>, ptr {{%.*}}, align 4
// CHECK: [[TOBOOL:%.*]] = icmp ne <4 x i32> [[COND]], zeroinitializer
// CHECK: select <4 x i1> [[TOBOOL]], <4 x i32> {{%.*}}, <4 x i32> {{%.*}}
int2x2 test_select_int_condition(int2x2 cond, int2x2 tVals, int2x2 fVals) {
  return select(cond, tVals, fVals);
}

// CHECK-LABEL: test_select_float_condition
// CHECK: [[COND:%.*]] = load <4 x float>, ptr {{%.*}}, align 4
// CHECK: [[TOBOOL:%.*]] = fcmp {{.*}}une <4 x float> [[COND]], zeroinitializer
// CHECK: select {{.*}}<4 x i1> [[TOBOOL]], <4 x float> {{%.*}}, <4 x float> {{%.*}}
float2x2 test_select_float_condition(float2x2 cond, float2x2 tVals,
                                          float2x2 fVals) {
  return select(cond, tVals, fVals);
}
