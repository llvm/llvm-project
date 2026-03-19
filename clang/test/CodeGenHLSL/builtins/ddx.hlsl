// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -fnative-half-type -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple spirv-pc-vulkan-pixel  %s \
// RUN:  -emit-llvm -disable-llvm-passes -fnative-half-type -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK-SPIRV

// CHECK-LABEL: define {{.*}} half @_ZN4hlsl8__detail8ddx_implIDhEET_S2_
// CHECK: %hlsl.ddx.coarse = call {{.*}} half @llvm.dx.ddx.coarse.f16(half %{{.*}})
// CHECK: ret half %hlsl.ddx.coarse
// CHECK-LABEL-SPIRV: half @_ZN4hlsl8__detail8ddx_implIDhEET_S2_
// CHECK-SPIRV: %spv.ddx = call {{.*}} half @llvm.spv.ddx.f16(half %{{.*}})
// CHECK-SPIRV: ret half %spv.ddx
half test_f16_ddx(half val) {
    return ddx(val);
}

// CHECK-LABEL: define {{.*}} <2 x half> @_ZN4hlsl8__detail8ddx_implIDv2_DhEET_S3_
// CHECK: %hlsl.ddx.coarse = call {{.*}} <2 x half> @llvm.dx.ddx.coarse.v2f16(<2 x half> %{{.*}})
// CHECK: ret <2 x half> %hlsl.ddx.coarse
// CHECK-LABEL-SPIRV: <2 x half> @_ZN4hlsl8__detail8ddx_implIDv2_DhEET_S3_
// CHECK-SPIRV: %spv.ddx = call {{.*}} <2 x half> @llvm.spv.ddx.v2f16(<2 x half> %{{.*}})
// CHECK-SPIRV: ret <2 x half> %spv.ddx
half2 test_f16_ddx2(half2 val) {
    return ddx(val);
}

// CHECK-LABEL: define {{.*}} <3 x half> @_ZN4hlsl8__detail8ddx_implIDv3_DhEET_S3_
// CHECK: %hlsl.ddx.coarse = call {{.*}} <3 x half> @llvm.dx.ddx.coarse.v3f16(<3 x half> %{{.*}})
// CHECK: ret <3 x half> %hlsl.ddx.coarse
// CHECK-LABEL-SPIRV: <3 x half> @_ZN4hlsl8__detail8ddx_implIDv3_DhEET_S3_
// CHECK-SPIRV: %spv.ddx = call {{.*}} <3 x half> @llvm.spv.ddx.v3f16(<3 x half> %{{.*}})
// CHECK-SPIRV: ret <3 x half> %spv.ddx
half3 test_f16_ddx3(half3 val) {
    return ddx(val);
}

// CHECK-LABEL: define {{.*}} <4 x half> @_ZN4hlsl8__detail8ddx_implIDv4_DhEET_S3_
// CHECK: %hlsl.ddx.coarse = call {{.*}} <4 x half> @llvm.dx.ddx.coarse.v4f16(<4 x half> %{{.*}})
// CHECK: ret <4 x half> %hlsl.ddx.coarse
// CHECK-LABEL-SPIRV: <4 x half> @_ZN4hlsl8__detail8ddx_implIDv4_DhEET_S3_
// CHECK-SPIRV: %spv.ddx = call {{.*}} <4 x half> @llvm.spv.ddx.v4f16(<4 x half> %{{.*}})
// CHECK-SPIRV: ret <4 x half> %spv.ddx
half4 test_f16_ddx4(half4 val) {
    return ddx(val);
}

// CHECK-LABEL: define {{.*}} float @_ZN4hlsl8__detail8ddx_implIfEET_S2_
// CHECK: %hlsl.ddx.coarse = call {{.*}} float @llvm.dx.ddx.coarse.f32(float %{{.*}})
// CHECK: ret float %hlsl.ddx.coarse
// CHECK-LABEL-SPIRV: float @_ZN4hlsl8__detail8ddx_implIfEET_S2_
// CHECK-SPIRV: %spv.ddx = call {{.*}} float @llvm.spv.ddx.f32(float %{{.*}})
// CHECK-SPIRV: ret float %spv.ddx
float test_f32_ddx(float val) {
    return ddx(val);
}

// CHECK-LABEL: define {{.*}} <2 x float> @_ZN4hlsl8__detail8ddx_implIDv2_fEET_S3_
// CHECK: %hlsl.ddx.coarse = call {{.*}} <2 x float> @llvm.dx.ddx.coarse.v2f32(<2 x float> %{{.*}})
// CHECK: ret <2 x float> %hlsl.ddx.coarse
// CHECK-LABEL-SPIRV: <2 x float> @_ZN4hlsl8__detail8ddx_implIDv2_fEET_S3_
// CHECK-SPIRV: %spv.ddx = call {{.*}} <2 x float> @llvm.spv.ddx.v2f32(<2 x float> %{{.*}})
// CHECK-SPIRV: ret <2 x float> %spv.ddx
float2 test_f32_ddx2(float2 val) {
    return ddx(val);
}

// CHECK-LABEL: define {{.*}} <3 x float> @_ZN4hlsl8__detail8ddx_implIDv3_fEET_S3_
// CHECK: %hlsl.ddx.coarse = call {{.*}} <3 x float> @llvm.dx.ddx.coarse.v3f32(<3 x float> %{{.*}})
// CHECK: ret <3 x float> %hlsl.ddx.coarse
// CHECK-LABEL-SPIRV: <3 x float> @_ZN4hlsl8__detail8ddx_implIDv3_fEET_S3_
// CHECK-SPIRV: %spv.ddx = call {{.*}} <3 x float> @llvm.spv.ddx.v3f32(<3 x float> %{{.*}})
// CHECK-SPIRV: ret <3 x float> %spv.ddx
float3 test_f32_ddx3(float3 val) {
    return ddx(val);
}

// CHECK-LABEL: define {{.*}} <4 x float> @_ZN4hlsl8__detail8ddx_implIDv4_fEET_S3_
// CHECK: %hlsl.ddx.coarse = call {{.*}} <4 x float> @llvm.dx.ddx.coarse.v4f32(<4 x float> %{{.*}})
// CHECK: ret <4 x float> %hlsl.ddx.coarse
// CHECK-LABEL-SPIRV: <4 x float> @_ZN4hlsl8__detail8ddx_implIDv4_fEET_S3_
// CHECK-SPIRV: %spv.ddx = call {{.*}} <4 x float> @llvm.spv.ddx.v4f32(<4 x float> %{{.*}})
// CHECK-SPIRV: ret <4 x float> %spv.ddx
float4 test_f32_ddx4(float4 val) {
    return ddx(val);
}
