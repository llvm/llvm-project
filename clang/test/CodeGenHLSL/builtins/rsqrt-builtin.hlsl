// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -fnative-int16-type -emit-llvm -disable-llvm-passes -o - | FileCheck %s


// CHECK-LABEL: builtin_rsqrt_half
// CHECK: %hlsl.rsqrt = call reassoc nnan ninf nsz arcp afn half @llvm.dx.rsqrt.f16(half %{{.*}})
// CHECK: ret half  %hlsl.rsqrt
half builtin_rsqrt_half(half p0) {
  return __builtin_hlsl_elementwise_rsqrt(p0);
}

// CHECK-LABEL: builtin_rsqrt_float
// CHECK: %hlsl.rsqrt = call reassoc nnan ninf nsz arcp afn float @llvm.dx.rsqrt.f32(float %{{.*}})
// CHECK: ret float  %hlsl.rsqrt
float builtin_rsqrt_float (float p0) {
  return __builtin_hlsl_elementwise_rsqrt(p0);
}
