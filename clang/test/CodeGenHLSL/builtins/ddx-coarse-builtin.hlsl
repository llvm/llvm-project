// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -fnative-half-type -o - | \
// RUN:  FileCheck %s -DTGT=dx
// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple spirv-pc-vulkan-compute  %s \
// RUN:  -emit-llvm -disable-llvm-passes -fnative-half-type -o - | \
// RUN:  FileCheck %s -DTGT=spv

// CHECK-LABEL: half @_Z19test_f16_ddx_coarseDh
// CHECK: %hlsl.ddx.coarse = call {{.*}} half @llvm.[[TGT]].ddx.coarse.f16(half %{{.*}})
// CHECK: ret half %hlsl.ddx.coarse
half test_f16_ddx_coarse(half val) {
    return __builtin_hlsl_elementwise_ddx_coarse(val);
}

// CHECK-LABEL: float @_Z19test_f32_ddx_coarsef
// CHECK: %hlsl.ddx.coarse = call {{.*}} float @llvm.[[TGT]].ddx.coarse.f32(float %{{.*}})
// CHECK: ret float %hlsl.ddx.coarse
float test_f32_ddx_coarse(float val) {
    return __builtin_hlsl_elementwise_ddx_coarse(val);
}
