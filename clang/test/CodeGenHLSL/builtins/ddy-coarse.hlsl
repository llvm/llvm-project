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
    return ddy_coarse(val);
}

// CHECK-LABEL: <2 x half> @_Z20test_f16_ddy_coarse2Dv2_Dh
// CHECK: %hlsl.ddy.coarse = call {{.*}} <2 x half> @llvm.dx.ddy.coarse.v2f16(<2 x half> %{{.*}})
// CHECK: ret <2 x half> %hlsl.ddy.coarse
// CHECK-LABEL-SPIRV: <2 x half> @_Z20test_f16_ddy_coarse2Dv2_Dh
// CHECK-SPIRV: %hlsl.ddy.coarse = call {{.*}} <2 x half> @llvm.spv.ddy.coarse.v2f16(<2 x half> %{{.*}})
// CHECK-SPIRV: ret <2 x half> %hlsl.ddy.coarse
half2 test_f16_ddy_coarse2(half2 val) {
    return ddy_coarse(val);
}

// CHECK-LABEL: <3 x half> @_Z20test_f16_ddy_coarse3Dv3_Dh
// CHECK: %hlsl.ddy.coarse = call {{.*}} <3 x half> @llvm.dx.ddy.coarse.v3f16(<3 x half> %{{.*}})
// CHECK: ret <3 x half> %hlsl.ddy.coarse
// CHECK-LABEL-SPIRV: <3 x half> @_Z20test_f16_ddy_coarse3Dv3_Dh
// CHECK-SPIRV: %hlsl.ddy.coarse = call {{.*}} <3 x half> @llvm.spv.ddy.coarse.v3f16(<3 x half> %{{.*}})
// CHECK-SPIRV: ret <3 x half> %hlsl.ddy.coarse
half3 test_f16_ddy_coarse3(half3 val) {
    return ddy_coarse(val);
}

// CHECK-LABEL: <4 x half> @_Z20test_f16_ddy_coarse4Dv4_Dh
// CHECK: %hlsl.ddy.coarse = call {{.*}} <4 x half> @llvm.dx.ddy.coarse.v4f16(<4 x half> %{{.*}})
// CHECK: ret <4 x half> %hlsl.ddy.coarse
// CHECK-LABEL-SPIRV: <4 x half> @_Z20test_f16_ddy_coarse4Dv4_Dh
// CHECK-SPIRV: %hlsl.ddy.coarse = call {{.*}} <4 x half> @llvm.spv.ddy.coarse.v4f16(<4 x half> %{{.*}})
// CHECK-SPIRV: ret <4 x half> %hlsl.ddy.coarse
half4 test_f16_ddy_coarse4(half4 val) {
    return ddy_coarse(val);
}

// CHECK-LABEL: float @_Z19test_f32_ddy_coarsef
// CHECK: %hlsl.ddy.coarse = call {{.*}} float @llvm.dx.ddy.coarse.f32(float %{{.*}})
// CHECK: ret float %hlsl.ddy.coarse
// CHECK-LABEL-SPIRV: float @_Z19test_f32_ddy_coarsef
// CHECK-SPIRV: %hlsl.ddy.coarse = call {{.*}} float @llvm.spv.ddy.coarse.f32(float %{{.*}})
// CHECK-SPIRV: ret float %hlsl.ddy.coarse
float test_f32_ddy_coarse(float val) {
    return ddy_coarse(val);
}

// CHECK-LABEL: <2 x float> @_Z20test_f32_ddy_coarse2Dv2_f
// CHECK: %hlsl.ddy.coarse = call {{.*}} <2 x float> @llvm.dx.ddy.coarse.v2f32(<2 x float> %{{.*}})
// CHECK: ret <2 x float> %hlsl.ddy.coarse
// CHECK-LABEL-SPIRV: <2 x float> @_Z20test_f32_ddy_coarse2Dv2_f
// CHECK-SPIRV: %hlsl.ddy.coarse = call {{.*}} <2 x float> @llvm.spv.ddy.coarse.v2f32(<2 x float> %{{.*}})
// CHECK-SPIRV: ret <2 x float> %hlsl.ddy.coarse
float2 test_f32_ddy_coarse2(float2 val) {
    return ddy_coarse(val);
}

// CHECK-LABEL: <3 x float> @_Z20test_f32_ddy_coarse3Dv3_f
// CHECK: %hlsl.ddy.coarse = call {{.*}} <3 x float> @llvm.dx.ddy.coarse.v3f32(<3 x float> %{{.*}})
// CHECK: ret <3 x float> %hlsl.ddy.coarse
// CHECK-LABEL-SPIRV: <3 x float> @_Z20test_f32_ddy_coarse3Dv3_f
// CHECK-SPIRV: %hlsl.ddy.coarse = call {{.*}} <3 x float> @llvm.spv.ddy.coarse.v3f32(<3 x float> %{{.*}})
// CHECK-SPIRV: ret <3 x float> %hlsl.ddy.coarse
float3 test_f32_ddy_coarse3(float3 val) {
    return ddy_coarse(val);
}

// CHECK-LABEL: <4 x float> @_Z20test_f32_ddy_coarse4Dv4_f
// CHECK: %hlsl.ddy.coarse = call {{.*}} <4 x float> @llvm.dx.ddy.coarse.v4f32(<4 x float> %{{.*}})
// CHECK: ret <4 x float> %hlsl.ddy.coarse
// CHECK-LABEL-SPIRV: <4 x float> @_Z20test_f32_ddy_coarse4Dv4_f
// CHECK-SPIRV: %hlsl.ddy.coarse = call {{.*}} <4 x float> @llvm.spv.ddy.coarse.v4f32(<4 x float> %{{.*}})
// CHECK-SPIRV: ret <4 x float> %hlsl.ddy.coarse
float4 test_f32_ddy_coarse4(float4 val) {
    return ddy_coarse(val);
}
