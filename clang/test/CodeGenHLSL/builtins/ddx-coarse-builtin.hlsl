// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -fnative-half-type -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple spirv-pc-vulkan-compute  %s \
// RUN:  -emit-llvm -disable-llvm-passes -fnative-half-type -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK-SPIRV

// CHECK-LABEL: half @_Z19test_f16_ddx_coarseDh
// CHECK: %hlsl.ddx.coarse = call {{.*}} half @llvm.dx.ddx.coarse.f16(half %{{.*}})
// CHECK: ret half %hlsl.ddx.coarse
// CHECK-LABEL-SPIRV: half @_Z19test_f16_ddx_coarseDh
// CHECK-SPIRV: %hlsl.ddx.coarse = call {{.*}} half @llvm.spv.ddx.coarse.f16(half %{{.*}})
// CHECK-SPIRV: ret half %hlsl.ddx.coarse
half test_f16_ddx_coarse(half val) {
    return __builtin_hlsl_elementwise_ddx_coarse(val);
}

// CHECK-LABEL: float @_Z19test_f32_ddx_coarsef
// CHECK: %hlsl.ddx.coarse = call {{.*}} float @llvm.dx.ddx.coarse.f32(float %{{.*}})
// CHECK: ret float %hlsl.ddx.coarse
// CHECK-LABEL-SPIRV: float @_Z19test_f32_ddx_coarsef
// CHECK-SPIRV: %hlsl.ddx.coarse = call {{.*}} float @llvm.spv.ddx.coarse.f32(float %{{.*}})
// CHECK-SPIRV: ret float %hlsl.ddx.coarse
float test_f32_ddx_coarse(float val) {
    return __builtin_hlsl_elementwise_ddx_coarse(val);
}
