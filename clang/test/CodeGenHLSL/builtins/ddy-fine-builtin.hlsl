// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -fnative-half-type -o - | \
// RUN:  FileCheck %s -DTGT=dx
// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple spirv-pc-vulkan-compute  %s \
// RUN:  -emit-llvm -disable-llvm-passes -fnative-half-type -o - | \
// RUN:  FileCheck %s -DTGT=spv

// CHECK-LABEL: half @_Z17test_f16_ddy_fineDh
// CHECK: %hlsl.ddy.fine = call {{.*}} half @llvm.[[TGT]].ddy.fine.f16(half %{{.*}})
// CHECK: ret half %hlsl.ddy.fine
half test_f16_ddy_fine(half val) {
    return __builtin_hlsl_elementwise_ddy_fine(val);
}

// CHECK-LABEL: float @_Z17test_f32_ddy_finef
// CHECK: %hlsl.ddy.fine = call {{.*}} float @llvm.[[TGT]].ddy.fine.f32(float %{{.*}})
// CHECK: ret float %hlsl.ddy.fine
float test_f32_ddy_fine(float val) {
    return __builtin_hlsl_elementwise_ddy_fine(val);
}
