// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -fnative-half-type -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple spirv-pc-vulkan-compute  %s \
// RUN:  -emit-llvm -disable-llvm-passes -fnative-half-type -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK-SPIRV

// CHECK-LABEL: half @_Z17test_f16_ddy_fineDh
// CHECK: %hlsl.ddy.fine = call {{.*}} half @llvm.dx.ddy.fine.f16(half %{{.*}})
// CHECK: ret half %hlsl.ddy.fine
// CHECK-LABEL-SPIRV: half @_Z17test_f16_ddy_fineDh
// CHECK-SPIRV: %hlsl.ddy.fine = call {{.*}} half @llvm.spv.ddy.fine.f16(half %{{.*}})
// CHECK-SPIRV: ret half %hlsl.ddy.fine
half test_f16_ddy_fine(half val) {
    return ddy_fine(val);
}

// CHECK-LABEL: <2 x half> @_Z18test_f16_ddy_fine2Dv2_Dh
// CHECK: %hlsl.ddy.fine = call {{.*}} <2 x half> @llvm.dx.ddy.fine.v2f16(<2 x half> %{{.*}})
// CHECK: ret <2 x half> %hlsl.ddy.fine
// CHECK-LABEL-SPIRV: <2 x half> @_Z18test_f16_ddy_fine2Dv2_Dh
// CHECK-SPIRV: %hlsl.ddy.fine = call {{.*}} <2 x half> @llvm.spv.ddy.fine.v2f16(<2 x half> %{{.*}})
// CHECK-SPIRV: ret <2 x half> %hlsl.ddy.fine
half2 test_f16_ddy_fine2(half2 val) {
    return ddy_fine(val);
}

// CHECK-LABEL: <3 x half> @_Z18test_f16_ddy_fine3Dv3_Dh
// CHECK: %hlsl.ddy.fine = call {{.*}} <3 x half> @llvm.dx.ddy.fine.v3f16(<3 x half> %{{.*}})
// CHECK: ret <3 x half> %hlsl.ddy.fine
// CHECK-LABEL-SPIRV: <3 x half> @_Z18test_f16_ddy_fine3Dv3_Dh
// CHECK-SPIRV: %hlsl.ddy.fine = call {{.*}} <3 x half> @llvm.spv.ddy.fine.v3f16(<3 x half> %{{.*}})
// CHECK-SPIRV: ret <3 x half> %hlsl.ddy.fine
half3 test_f16_ddy_fine3(half3 val) {
    return ddy_fine(val);
}

// CHECK-LABEL: <4 x half> @_Z18test_f16_ddy_fine4Dv4_Dh
// CHECK: %hlsl.ddy.fine = call {{.*}} <4 x half> @llvm.dx.ddy.fine.v4f16(<4 x half> %{{.*}})
// CHECK: ret <4 x half> %hlsl.ddy.fine
// CHECK-LABEL-SPIRV: <4 x half> @_Z18test_f16_ddy_fine4Dv4_Dh
// CHECK-SPIRV: %hlsl.ddy.fine = call {{.*}} <4 x half> @llvm.spv.ddy.fine.v4f16(<4 x half> %{{.*}})
// CHECK-SPIRV: ret <4 x half> %hlsl.ddy.fine
half4 test_f16_ddy_fine4(half4 val) {
    return ddy_fine(val);
}

// CHECK-LABEL: float @_Z17test_f32_ddy_finef
// CHECK: %hlsl.ddy.fine = call {{.*}} float @llvm.dx.ddy.fine.f32(float %{{.*}})
// CHECK: ret float %hlsl.ddy.fine
// CHECK-LABEL-SPIRV: float @_Z17test_f32_ddy_finef
// CHECK-SPIRV: %hlsl.ddy.fine = call {{.*}} float @llvm.spv.ddy.fine.f32(float %{{.*}})
// CHECK-SPIRV: ret float %hlsl.ddy.fine
float test_f32_ddy_fine(float val) {
    return ddy_fine(val);
}

// CHECK-LABEL: <2 x float> @_Z18test_f32_ddy_fine2Dv2_f
// CHECK: %hlsl.ddy.fine = call {{.*}} <2 x float> @llvm.dx.ddy.fine.v2f32(<2 x float> %{{.*}})
// CHECK: ret <2 x float> %hlsl.ddy.fine
// CHECK-LABEL-SPIRV: <2 x float> @_Z18test_f32_ddy_fine2Dv2_f
// CHECK-SPIRV: %hlsl.ddy.fine = call {{.*}} <2 x float> @llvm.spv.ddy.fine.v2f32(<2 x float> %{{.*}})
// CHECK-SPIRV: ret <2 x float> %hlsl.ddy.fine
float2 test_f32_ddy_fine2(float2 val) {
    return ddy_fine(val);
}

// CHECK-LABEL: <3 x float> @_Z18test_f32_ddy_fine3Dv3_f
// CHECK: %hlsl.ddy.fine = call {{.*}} <3 x float> @llvm.dx.ddy.fine.v3f32(<3 x float> %{{.*}})
// CHECK: ret <3 x float> %hlsl.ddy.fine
// CHECK-LABEL-SPIRV: <3 x float> @_Z18test_f32_ddy_fine3Dv3_f
// CHECK-SPIRV: %hlsl.ddy.fine = call {{.*}} <3 x float> @llvm.spv.ddy.fine.v3f32(<3 x float> %{{.*}})
// CHECK-SPIRV: ret <3 x float> %hlsl.ddy.fine
float3 test_f32_ddy_fine3(float3 val) {
    return ddy_fine(val);
}

// CHECK-LABEL: <4 x float> @_Z18test_f32_ddy_fine4Dv4_f
// CHECK: %hlsl.ddy.fine = call {{.*}} <4 x float> @llvm.dx.ddy.fine.v4f32(<4 x float> %{{.*}})
// CHECK: ret <4 x float> %hlsl.ddy.fine
// CHECK-LABEL-SPIRV: <4 x float> @_Z18test_f32_ddy_fine4Dv4_f
// CHECK-SPIRV: %hlsl.ddy.fine = call {{.*}} <4 x float> @llvm.spv.ddy.fine.v4f32(<4 x float> %{{.*}})
// CHECK-SPIRV: ret <4 x float> %hlsl.ddy.fine
float4 test_f32_ddy_fine4(float4 val) {
    return ddy_fine(val);
}
