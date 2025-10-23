// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -fnative-half-type -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK

// CHECK: define hidden noundef nofpclass(nan inf) half @
// CHECK: %hlsl.ddy.coarse = call reassoc nnan ninf nsz arcp afn half @llvm.dx.ddy.coarse.f16(half %{{.*}})
// CHECK: ret half %hlsl.ddy.coarse
half test_f16_ddy_coarse(half val) {
    return __builtin_hlsl_elementwise_ddy_coarse(val);
}

// CHECK: define hidden noundef nofpclass(nan inf) float @
// CHECK: %hlsl.ddy.coarse = call reassoc nnan ninf nsz arcp afn float @llvm.dx.ddy.coarse.f32(float %{{.*}})
// CHECK: ret float %hlsl.ddy.coarse
float test_f32_ddy_coarse(float val) {
    return __builtin_hlsl_elementwise_ddy_coarse(val);
}