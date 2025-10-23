// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -fnative-half-type -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK

// CHECK: define hidden noundef nofpclass(nan inf) half @
// CHECK: %hlsl.ddx.coarse = call reassoc nnan ninf nsz arcp afn half @llvm.dx.ddx.coarse.f16(half %{{.*}})
// CHECK: ret half %hlsl.ddx.coarse
half test_f16_ddx_coarse(half val) {
    return __builtin_hlsl_elementwise_ddx_coarse(val);
}

// CHECK: define hidden noundef nofpclass(nan inf) float @
// CHECK: %hlsl.ddx.coarse = call reassoc nnan ninf nsz arcp afn float @llvm.dx.ddx.coarse.f32(float %{{.*}})
// CHECK: ret float %hlsl.ddx.coarse
float test_f32_ddx_coarse(float val) {
    return __builtin_hlsl_elementwise_ddx_coarse(val);
}