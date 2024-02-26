// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -emit-llvm -disable-llvm-passes -o - | FileCheck %s



// CHECK-LABEL: builtin_lerp_half_scalar
// CHECK: %3 = fsub double %conv1, %conv
// CHECK: %4 = fmul double %conv2, %3
// CHECK: %dx.lerp = fadd double %conv, %4
// CHECK: %conv3 = fptrunc double %dx.lerp to half
// CHECK: ret half %conv3
half builtin_lerp_half_scalar (half p0) {
  return __builtin_hlsl_lerp ( p0, p0, p0 );
}

// CHECK-LABEL: builtin_lerp_float_scalar
// CHECK: %3 = fsub double %conv1, %conv
// CHECK: %4 = fmul double %conv2, %3
// CHECK: %dx.lerp = fadd double %conv, %4
// CHECK: %conv3 = fptrunc double %dx.lerp to float
// CHECK: ret float %conv3
float builtin_lerp_float_scalar ( float p0) {
  return __builtin_hlsl_lerp ( p0, p0, p0 );
}

// CHECK-LABEL: builtin_lerp_half_vector
// CHECK: %dx.lerp = call <3 x half> @llvm.dx.lerp.v3f16(<3 x half> %0, <3 x half> %1, <3 x half> %2)
// CHECK: ret <3 x half> %dx.lerp
half3 builtin_lerp_half_vector (half3 p0) {
  return __builtin_hlsl_lerp ( p0, p0, p0 );
}

// CHECK-LABEL: builtin_lerp_floar_vector
// CHECK: %dx.lerp = call <2 x float> @llvm.dx.lerp.v2f32(<2 x float> %0, <2 x float> %1, <2 x float> %2)
// CHECK: ret <2 x float> %dx.lerp
float2 builtin_lerp_floar_vector ( float2 p0) {
  return __builtin_hlsl_lerp ( p0, p0, p0 );
}
