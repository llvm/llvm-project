// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -emit-llvm -disable-llvm-passes -o - | FileCheck %s


// CHECK-LABEL: builtin_lerp_half
// CHECK: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn half @llvm.dx.lerp.f16(half %{{.*}}, half %{{.*}}, half %{{.*}})
// CHECK: ret half %hlsl.lerp
half builtin_lerp_half(half p0) { return __builtin_hlsl_lerp(p0, p0, p0); }

// CHECK-LABEL: builtin_lerp_float
// CHECK: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn float @llvm.dx.lerp.f32(float %{{.*}}, float %{{.*}}, float %{{.*}})
// CHECK: ret float %hlsl.lerp
float builtin_lerp_float(float p0) { return __builtin_hlsl_lerp(p0, p0, p0); }
