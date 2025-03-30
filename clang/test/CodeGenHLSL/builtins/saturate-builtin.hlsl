// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -emit-llvm -disable-llvm-passes -o - | FileCheck %s


// CHECK-LABEL: builtin_saturate_half
// CHECK: %hlsl.saturate = call reassoc nnan ninf nsz arcp afn half @llvm.dx.saturate.f16(half %{{.*}})
// CHECK: ret half  %hlsl.saturate
half builtin_saturate_half(half p0) {
  return __builtin_hlsl_elementwise_saturate(p0);
}

// CHECK-LABEL: builtin_saturate_float
// CHECK: %hlsl.saturate = call reassoc nnan ninf nsz arcp afn float @llvm.dx.saturate.f32(float %{{.*}})
// CHECK: ret float  %hlsl.saturate
float builtin_saturate_float (float p0) {
  return __builtin_hlsl_elementwise_saturate(p0);
}
