// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -fnative-int16-type -emit-llvm -disable-llvm-passes -o - | FileCheck %s


// CHECK-LABEL: builtin_normalize_half
// CHECK: %hlsl.normalize = call reassoc nnan ninf nsz arcp afn half @llvm.dx.normalize.f16(half %{{.*}})
// CHECK: ret half  %hlsl.normalize
half builtin_normalize_half(half p0) {
  return __builtin_hlsl_normalize(p0);
}

// CHECK-LABEL: builtin_normalize_float
// CHECK: %hlsl.normalize = call reassoc nnan ninf nsz arcp afn float @llvm.dx.normalize.f32(float %{{.*}})
// CHECK: ret float  %hlsl.normalize
float builtin_normalize_float (float p0) {
  return __builtin_hlsl_normalize(p0);
}
