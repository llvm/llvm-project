// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -emit-llvm -disable-llvm-passes -o - | FileCheck %s


// CHECK-LABEL: builtin_frac_half
// CHECK: %hlsl.frac = call reassoc nnan ninf nsz arcp afn half @llvm.dx.frac.f16(half %{{.*}})
// CHECK: ret half  %hlsl.frac
half builtin_frac_half(half p0) {
  return __builtin_hlsl_elementwise_frac(p0);
}

// CHECK-LABEL: builtin_frac_float
// CHECK: %hlsl.frac = call reassoc nnan ninf nsz arcp afn float @llvm.dx.frac.f32(float %{{.*}})
// CHECK: ret float  %hlsl.frac
float builtin_frac_float (float p0) {
  return __builtin_hlsl_elementwise_frac(p0);
}
