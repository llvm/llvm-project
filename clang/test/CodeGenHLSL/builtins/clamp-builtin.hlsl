// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -fnative-int16-type -emit-llvm -disable-llvm-passes -o - | FileCheck %s


// CHECK-LABEL: builtin_clamp_half
// CHECK: %hlsl.clamp = call reassoc nnan ninf nsz arcp afn half @llvm.dx.nclamp.f16(half %{{.*}}, half %{{.*}}, half %{{.*}})
// CHECK: ret half %hlsl.clamp
half builtin_clamp_half(half p0) { return __builtin_hlsl_elementwise_clamp(p0, p0, p0); }

// CHECK-LABEL: builtin_clamp_float
// CHECK: %hlsl.clamp = call reassoc nnan ninf nsz arcp afn float @llvm.dx.nclamp.f32(float %{{.*}}, float %{{.*}}, float %{{.*}})
// CHECK: ret float %hlsl.clamp
float builtin_clamp_float(float p0) { return __builtin_hlsl_elementwise_clamp(p0, p0, p0); }
