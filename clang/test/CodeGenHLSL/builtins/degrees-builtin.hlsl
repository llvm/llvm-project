// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -fnative-int16-type -emit-llvm -disable-llvm-passes -o - | FileCheck %s


// CHECK-LABEL: builtin_degrees_half
// CHECK: %hlsl.degrees = call reassoc nnan ninf nsz arcp afn half @llvm.dx.degrees.f16(half %{{.*}})
// CHECK: ret half  %hlsl.degrees
half builtin_degrees_half(half p0) {
  return __builtin_hlsl_elementwise_degrees(p0);
}

// CHECK-LABEL: builtin_degrees_float
// CHECK: %hlsl.degrees = call reassoc nnan ninf nsz arcp afn float @llvm.dx.degrees.f32(float %{{.*}})
// CHECK: ret float  %hlsl.degrees
float builtin_degrees_float (float p0) {
  return __builtin_hlsl_elementwise_degrees(p0);
}
