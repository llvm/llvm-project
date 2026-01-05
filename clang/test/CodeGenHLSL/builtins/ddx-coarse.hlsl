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
    return ddx_coarse(val);
}

// CHECK-LABEL: <2 x half> @_Z20test_f16_ddx_coarse2Dv2_Dh
// CHECK: %hlsl.ddx.coarse = call {{.*}} <2 x half> @llvm.dx.ddx.coarse.v2f16(<2 x half> %{{.*}})
// CHECK: ret <2 x half> %hlsl.ddx.coarse
// CHECK-LABEL-SPIRV: <2 x half> @_Z20test_f16_ddx_coarse2Dv2_Dh
// CHECK-SPIRV: %hlsl.ddx.coarse = call {{.*}} <2 x half> @llvm.spv.ddx.coarse.v2f16(<2 x half> %{{.*}})
// CHECK-SPIRV: ret <2 x half> %hlsl.ddx.coarse
half2 test_f16_ddx_coarse2(half2 val) {
    return ddx_coarse(val);
}

// CHECK-LABEL: <3 x half> @_Z20test_f16_ddx_coarse3Dv3_Dh
// CHECK: %hlsl.ddx.coarse = call {{.*}} <3 x half> @llvm.dx.ddx.coarse.v3f16(<3 x half> %{{.*}})
// CHECK: ret <3 x half> %hlsl.ddx.coarse
// CHECK-LABEL-SPIRV: <3 x half> @_Z20test_f16_ddx_coarse3Dv3_Dh
// CHECK-SPIRV: %hlsl.ddx.coarse = call {{.*}} <3 x half> @llvm.spv.ddx.coarse.v3f16(<3 x half> %{{.*}})
// CHECK-SPIRV: ret <3 x half> %hlsl.ddx.coarse
half3 test_f16_ddx_coarse3(half3 val) {
    return ddx_coarse(val);
}

// CHECK-LABEL: <4 x half> @_Z20test_f16_ddx_coarse4Dv4_Dh
// CHECK: %hlsl.ddx.coarse = call {{.*}} <4 x half> @llvm.dx.ddx.coarse.v4f16(<4 x half> %{{.*}})
// CHECK: ret <4 x half> %hlsl.ddx.coarse
// CHECK-LABEL-SPIRV: <4 x half> @_Z20test_f16_ddx_coarse4Dv4_Dh
// CHECK-SPIRV: %hlsl.ddx.coarse = call {{.*}} <4 x half> @llvm.spv.ddx.coarse.v4f16(<4 x half> %{{.*}})
// CHECK-SPIRV: ret <4 x half> %hlsl.ddx.coarse
half4 test_f16_ddx_coarse4(half4 val) {
    return ddx_coarse(val);
}

// CHECK-LABEL: float @_Z19test_f32_ddx_coarsef
// CHECK: %hlsl.ddx.coarse = call {{.*}} float @llvm.dx.ddx.coarse.f32(float %{{.*}})
// CHECK: ret float %hlsl.ddx.coarse
// CHECK-LABEL-SPIRV: float @_Z19test_f32_ddx_coarsef
// CHECK-SPIRV: %hlsl.ddx.coarse = call {{.*}} float @llvm.spv.ddx.coarse.f32(float %{{.*}})
// CHECK-SPIRV: ret float %hlsl.ddx.coarse
float test_f32_ddx_coarse(float val) {
    return ddx_coarse(val);
}

// CHECK-LABEL: <2 x float> @_Z20test_f32_ddx_coarse2Dv2_f
// CHECK: %hlsl.ddx.coarse = call {{.*}} <2 x float> @llvm.dx.ddx.coarse.v2f32(<2 x float> %{{.*}})
// CHECK: ret <2 x float> %hlsl.ddx.coarse
// CHECK-LABEL-SPIRV: <2 x float> @_Z20test_f32_ddx_coarse2Dv2_f
// CHECK-SPIRV: %hlsl.ddx.coarse = call {{.*}} <2 x float> @llvm.spv.ddx.coarse.v2f32(<2 x float> %{{.*}})
// CHECK-SPIRV: ret <2 x float> %hlsl.ddx.coarse
float2 test_f32_ddx_coarse2(float2 val) {
    return ddx_coarse(val);
}

// CHECK-LABEL: <3 x float> @_Z20test_f32_ddx_coarse3Dv3_f
// CHECK: %hlsl.ddx.coarse = call {{.*}} <3 x float> @llvm.dx.ddx.coarse.v3f32(<3 x float> %{{.*}})
// CHECK: ret <3 x float> %hlsl.ddx.coarse
// CHECK-LABEL-SPIRV: <3 x float> @_Z20test_f32_ddx_coarse3Dv3_f
// CHECK-SPIRV: %hlsl.ddx.coarse = call {{.*}} <3 x float> @llvm.spv.ddx.coarse.v3f32(<3 x float> %{{.*}})
// CHECK-SPIRV: ret <3 x float> %hlsl.ddx.coarse
float3 test_f32_ddx_coarse3(float3 val) {
    return ddx_coarse(val);
}

// CHECK-LABEL: <4 x float> @_Z20test_f32_ddx_coarse4Dv4_f
// CHECK: %hlsl.ddx.coarse = call {{.*}} <4 x float> @llvm.dx.ddx.coarse.v4f32(<4 x float> %{{.*}})
// CHECK: ret <4 x float> %hlsl.ddx.coarse
// CHECK-LABEL-SPIRV: <4 x float> @_Z20test_f32_ddx_coarse4Dv4_f
// CHECK-SPIRV: %hlsl.ddx.coarse = call {{.*}} <4 x float> @llvm.spv.ddx.coarse.v4f32(<4 x float> %{{.*}})
// CHECK-SPIRV: ret <4 x float> %hlsl.ddx.coarse
float4 test_f32_ddx_coarse4(float4 val) {
    return ddx_coarse(val);
}
