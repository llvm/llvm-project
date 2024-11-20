// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// CHECK-LABEL: builtin_test_clamp_int4
// CHECK: %hlsl.clamp = call <4 x i32> @llvm.dx.sclamp.v4i32(<4 x i32> %0, <4 x i32> %1, <4 x i32> %2)
// CHECK: ret <4 x i32> %hlsl.clamp
int4 builtin_test_clamp_int4(int4 p0, int4 p1, int4 p2) {
  return __builtin_hlsl_elementwise_clamp(p0, p1, p2);
}
