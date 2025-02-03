// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// CHECK-LABEL: builtin_lerp_half_vector
// CHECK: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn <3 x half> @llvm.dx.lerp.v3f16(<3 x half> %0, <3 x half> %1, <3 x half> %2)
// CHECK: ret <3 x half> %hlsl.lerp
half3 builtin_lerp_half_vector (half3 p0) {
  return __builtin_hlsl_lerp ( p0, p0, p0 );
}

// CHECK-LABEL: builtin_lerp_floar_vector
// CHECK: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.dx.lerp.v2f32(<2 x float> %0, <2 x float> %1, <2 x float> %2)
// CHECK: ret <2 x float> %hlsl.lerp
float2 builtin_lerp_floar_vector ( float2 p0) {
  return __builtin_hlsl_lerp ( p0, p0, p0 );
}
