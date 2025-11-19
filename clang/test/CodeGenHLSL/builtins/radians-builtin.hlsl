// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -fnative-int16-type -emit-llvm -disable-llvm-passes -o - | FileCheck %s


// CHECK-LABEL: builtin_radians_half
// CHECK: %hlsl.radians = call reassoc nnan ninf nsz arcp afn half @llvm.dx.radians.f16(half %{{.*}})
// CHECK: ret half  %hlsl.radians
half builtin_radians_half(half p0) {
  return __builtin_hlsl_elementwise_radians(p0);
}

// CHECK-LABEL: builtin_radians_float
// CHECK: %hlsl.radians = call reassoc nnan ninf nsz arcp afn float @llvm.dx.radians.f32(float %{{.*}})
// CHECK: ret float  %hlsl.radians
float builtin_radians_float (float p0) {
  return __builtin_hlsl_elementwise_radians(p0);
}
