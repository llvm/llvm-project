// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// // CHECK-LABEL: builtin_test_asuint_float
// // CHECK: bitcast float %0 to i32
// // CHECK: ret <4 x i32> %dx.clamp
// export uint builtin_test_asuint_float(float p0) {
//   return __builtin_hlsl_elementwise_asuint(p0);
// }


// // CHECK-LABEL: builtin_test_asuint_float
// // CHECK: bitcast float %0 to i32
// // CHECK: ret <4 x i32> %dx.clamp
// export uint builtin_test_asuint_double(double p0) {
//   return __builtin_hlsl_elementwise_asuint(p0);
// }


// // CHECK-LABEL: builtin_test_asuint_float
// // CHECK: bitcast float %0 to i32
// // CHECK: ret <4 x i32> %dx.clamp
// export uint builtin_test_asuint_half(half p0) {
//   return __builtin_hlsl_elementwise_asuint(p0);
// }


// // CHECK-LABEL: builtin_test_asuint_float
// // CHECK: bitcast float %0 to i32
// // CHECK: ret <4 x i32> %dx.clamp
// export uint4 builtin_test_asuint_float_vector(float p0) {
//   return __builtin_hlsl_elementwise_asuint(p0);
// }


// CHECK-LABEL: builtin_test_asuint_float
// CHECK: bitcast float %0 to i32
// CHECK: ret <4 x i32> %dx.clamp
export uint4 builtin_test_asuint_floa4t(float p0) {
  return asuint(p0);
}

// export uint4 builtin_test_asuint4_uint(uint p0) {
//   return __builtin_hlsl_elementwise_asuint(p0);
// }


// export uint4 builtin_test_asuint4_int(int p0) {
//   return __builtin_hlsl_elementwise_asuint(p0);
// }

// export uint builtin_test_asuint_float(float p0) {
//   return __builtin_hlsl_elementwise_asuint(p0);
// }