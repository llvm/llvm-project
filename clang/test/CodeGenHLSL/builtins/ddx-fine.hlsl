// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -fnative-half-type -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple spirv-pc-vulkan-compute  %s \
// RUN:  -emit-llvm -disable-llvm-passes -fnative-half-type -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK-SPIRV

// CHECK-LABEL: half @_Z17test_f16_ddx_fineDh
// CHECK: %hlsl.ddx.fine = call {{.*}} half @llvm.dx.ddx.fine.f16(half %{{.*}})
// CHECK: ret half %hlsl.ddx.fine
// CHECK-LABEL-SPIRV: half @_Z17test_f16_ddx_fineDh
// CHECK-SPIRV: %hlsl.ddx.fine = call {{.*}} half @llvm.spv.ddx.fine.f16(half %{{.*}})
// CHECK-SPIRV: ret half %hlsl.ddx.fine
half test_f16_ddx_fine(half val) {
    return ddx_fine(val);
}

// CHECK-LABEL: <2 x half> @_Z18test_f16_ddx_fine2Dv2_Dh
// CHECK: %hlsl.ddx.fine = call {{.*}} <2 x half> @llvm.dx.ddx.fine.v2f16(<2 x half> %{{.*}})
// CHECK: ret <2 x half> %hlsl.ddx.fine
// CHECK-LABEL-SPIRV: <2 x half> @_Z18test_f16_ddx_fine2Dv2_Dh
// CHECK-SPIRV: %hlsl.ddx.fine = call {{.*}} <2 x half> @llvm.spv.ddx.fine.v2f16(<2 x half> %{{.*}})
// CHECK-SPIRV: ret <2 x half> %hlsl.ddx.fine
half2 test_f16_ddx_fine2(half2 val) {
    return ddx_fine(val);
}

// CHECK-LABEL: <3 x half> @_Z18test_f16_ddx_fine3Dv3_Dh
// CHECK: %hlsl.ddx.fine = call {{.*}} <3 x half> @llvm.dx.ddx.fine.v3f16(<3 x half> %{{.*}})
// CHECK: ret <3 x half> %hlsl.ddx.fine
// CHECK-LABEL-SPIRV: <3 x half> @_Z18test_f16_ddx_fine3Dv3_Dh
// CHECK-SPIRV: %hlsl.ddx.fine = call {{.*}} <3 x half> @llvm.spv.ddx.fine.v3f16(<3 x half> %{{.*}})
// CHECK-SPIRV: ret <3 x half> %hlsl.ddx.fine
half3 test_f16_ddx_fine3(half3 val) {
    return ddx_fine(val);
}

// CHECK-LABEL: <4 x half> @_Z18test_f16_ddx_fine4Dv4_Dh
// CHECK: %hlsl.ddx.fine = call {{.*}} <4 x half> @llvm.dx.ddx.fine.v4f16(<4 x half> %{{.*}})
// CHECK: ret <4 x half> %hlsl.ddx.fine
// CHECK-LABEL-SPIRV: <4 x half> @_Z18test_f16_ddx_fine4Dv4_Dh
// CHECK-SPIRV: %hlsl.ddx.fine = call {{.*}} <4 x half> @llvm.spv.ddx.fine.v4f16(<4 x half> %{{.*}})
// CHECK-SPIRV: ret <4 x half> %hlsl.ddx.fine
half4 test_f16_ddx_fine4(half4 val) {
    return ddx_fine(val);
}

// CHECK-LABEL: float @_Z17test_f32_ddx_finef
// CHECK: %hlsl.ddx.fine = call {{.*}} float @llvm.dx.ddx.fine.f32(float %{{.*}})
// CHECK: ret float %hlsl.ddx.fine
// CHECK-LABEL-SPIRV: float @_Z17test_f32_ddx_finef
// CHECK-SPIRV: %hlsl.ddx.fine = call {{.*}} float @llvm.spv.ddx.fine.f32(float %{{.*}})
// CHECK-SPIRV: ret float %hlsl.ddx.fine
float test_f32_ddx_fine(float val) {
    return ddx_fine(val);
}

// CHECK-LABEL: <2 x float> @_Z18test_f32_ddx_fine2Dv2_f
// CHECK: %hlsl.ddx.fine = call {{.*}} <2 x float> @llvm.dx.ddx.fine.v2f32(<2 x float> %{{.*}})
// CHECK: ret <2 x float> %hlsl.ddx.fine
// CHECK-LABEL-SPIRV: <2 x float> @_Z18test_f32_ddx_fine2Dv2_f
// CHECK-SPIRV: %hlsl.ddx.fine = call {{.*}} <2 x float> @llvm.spv.ddx.fine.v2f32(<2 x float> %{{.*}})
// CHECK-SPIRV: ret <2 x float> %hlsl.ddx.fine
float2 test_f32_ddx_fine2(float2 val) {
    return ddx_fine(val);
}

// CHECK-LABEL: <3 x float> @_Z18test_f32_ddx_fine3Dv3_f
// CHECK: %hlsl.ddx.fine = call {{.*}} <3 x float> @llvm.dx.ddx.fine.v3f32(<3 x float> %{{.*}})
// CHECK: ret <3 x float> %hlsl.ddx.fine
// CHECK-LABEL-SPIRV: <3 x float> @_Z18test_f32_ddx_fine3Dv3_f
// CHECK-SPIRV: %hlsl.ddx.fine = call {{.*}} <3 x float> @llvm.spv.ddx.fine.v3f32(<3 x float> %{{.*}})
// CHECK-SPIRV: ret <3 x float> %hlsl.ddx.fine
float3 test_f32_ddx_fine3(float3 val) {
    return ddx_fine(val);
}

// CHECK-LABEL: <4 x float> @_Z18test_f32_ddx_fine4Dv4_f
// CHECK: %hlsl.ddx.fine = call {{.*}} <4 x float> @llvm.dx.ddx.fine.v4f32(<4 x float> %{{.*}})
// CHECK: ret <4 x float> %hlsl.ddx.fine
// CHECK-LABEL-SPIRV: <4 x float> @_Z18test_f32_ddx_fine4Dv4_f
// CHECK-SPIRV: %hlsl.ddx.fine = call {{.*}} <4 x float> @llvm.spv.ddx.fine.v4f32(<4 x float> %{{.*}})
// CHECK-SPIRV: ret <4 x float> %hlsl.ddx.fine
float4 test_f32_ddx_fine4(float4 val) {
    return ddx_fine(val);
}
