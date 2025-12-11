// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -fnative-half-type -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple spirv-pc-vulkan-compute  %s \
// RUN:  -emit-llvm -disable-llvm-passes -fnative-half-type -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK-SPIRV

// CHECK-LABEL: half @_Z19test_f16_ddy_coarseDh
// CHECK: %hlsl.ddy.coarse = call {{.*}} half @llvm.dx.ddy.coarse.f16(half %{{.*}})
// CHECK: ret half %hlsl.ddy.coarse
// CHECK-LABEL-SPIRV: half @_Z19test_f16_ddy_coarseDh
// CHECK-SPIRV: %hlsl.ddy.coarse = call {{.*}} half @llvm.spv.ddy.coarse.f16(half %{{.*}})
// CHECK-SPIRV: ret half %hlsl.ddy.coarse
half test_f16_ddy_coarse(half val) {
    return __builtin_hlsl_elementwise_ddy_coarse(val);
}

// CHECK-LABEL: float @_Z19test_f32_ddy_coarsef
// CHECK: %hlsl.ddy.coarse = call {{.*}} float @llvm.dx.ddy.coarse.f32(float %{{.*}})
// CHECK: ret float %hlsl.ddy.coarse
// CHECK-LABEL-SPIRV: float @_Z19test_f32_ddy_coarsef
// CHECK-SPIRV: %hlsl.ddy.coarse = call {{.*}} float @llvm.spv.ddy.coarse.f32(float %{{.*}})
// CHECK-SPIRV: ret float %hlsl.ddy.coarse
float test_f32_ddy_coarse(float val) {
    return __builtin_hlsl_elementwise_ddy_coarse(val);
}
