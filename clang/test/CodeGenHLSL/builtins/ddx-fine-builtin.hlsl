// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -fnative-half-type -o - | \
// RUN:  FileCheck %s -DTGT=dx
// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple spirv-pc-vulkan-compute  %s \
// RUN:  -emit-llvm -disable-llvm-passes -fnative-half-type -o - | \
// RUN:  FileCheck %s -DTGT=spv

// CHECK-LABEL: half @_Z17test_f16_ddx_fineDh
// CHECK: %hlsl.ddx.fine = call {{.*}} half @llvm.[[TGT]].ddx.fine.f16(half %{{.*}})
// CHECK: ret half %hlsl.ddx.fine
half test_f16_ddx_fine(half val) {
    return __builtin_hlsl_elementwise_ddx_fine(val);
}

// CHECK-LABEL: float @_Z17test_f32_ddx_finef
// CHECK: %hlsl.ddx.fine = call {{.*}} float @llvm.[[TGT]].ddx.fine.f32(float %{{.*}})
// CHECK: ret float %hlsl.ddx.fine
float test_f32_ddx_fine(float val) {
    return __builtin_hlsl_elementwise_ddx_fine(val);
}
