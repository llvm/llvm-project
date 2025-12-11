// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -fnative-half-type -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple spirv-pc-vulkan-pixel  %s \
// RUN:  -emit-llvm -disable-llvm-passes -fnative-half-type -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK-SPIRV

// CHECK-LABEL: define {{.*}} half @_ZN4hlsl8__detail8ddy_implIDhEET_S2_
// CHECK: %hlsl.ddy.coarse = call {{.*}} half @llvm.dx.ddy.coarse.f16(half %{{.*}})
// CHECK: ret half %hlsl.ddy.coarse
// CHECK-LABEL-SPIRV: half @_ZN4hlsl8__detail8ddy_implIDhEET_S2_
// CHECK-SPIRV: %spv.ddy = call {{.*}} half @llvm.spv.ddy.f16(half %{{.*}})
// CHECK-SPIRV: ret half %spv.ddy
half test_f16_ddy(half val) {
    return ddy(val);
}

// CHECK-LABEL: define {{.*}} <2 x half> @_ZN4hlsl8__detail8ddy_implIDv2_DhEET_S3_
// CHECK: %hlsl.ddy.coarse = call {{.*}} <2 x half> @llvm.dx.ddy.coarse.v2f16(<2 x half> %{{.*}})
// CHECK: ret <2 x half> %hlsl.ddy.coarse
// CHECK-LABEL-SPIRV: <2 x half> @_ZN4hlsl8__detail8ddy_implIDv2_DhEET_S3_
// CHECK-SPIRV: %spv.ddy = call {{.*}} <2 x half> @llvm.spv.ddy.v2f16(<2 x half> %{{.*}})
// CHECK-SPIRV: ret <2 x half> %spv.ddy
half2 test_f16_ddy2(half2 val) {
    return ddy(val);
}

// CHECK-LABEL: define {{.*}} <3 x half> @_ZN4hlsl8__detail8ddy_implIDv3_DhEET_S3_
// CHECK: %hlsl.ddy.coarse = call {{.*}} <3 x half> @llvm.dx.ddy.coarse.v3f16(<3 x half> %{{.*}})
// CHECK: ret <3 x half> %hlsl.ddy.coarse
// CHECK-LABEL-SPIRV: <3 x half> @_ZN4hlsl8__detail8ddy_implIDv3_DhEET_S3_
// CHECK-SPIRV: %spv.ddy = call {{.*}} <3 x half> @llvm.spv.ddy.v3f16(<3 x half> %{{.*}})
// CHECK-SPIRV: ret <3 x half> %spv.ddy
half3 test_f16_ddy3(half3 val) {
    return ddy(val);
}

// CHECK-LABEL: define {{.*}} <4 x half> @_ZN4hlsl8__detail8ddy_implIDv4_DhEET_S3_
// CHECK: %hlsl.ddy.coarse = call {{.*}} <4 x half> @llvm.dx.ddy.coarse.v4f16(<4 x half> %{{.*}})
// CHECK: ret <4 x half> %hlsl.ddy.coarse
// CHECK-LABEL-SPIRV: <4 x half> @_ZN4hlsl8__detail8ddy_implIDv4_DhEET_S3_
// CHECK-SPIRV: %spv.ddy = call {{.*}} <4 x half> @llvm.spv.ddy.v4f16(<4 x half> %{{.*}})
// CHECK-SPIRV: ret <4 x half> %spv.ddy
half4 test_f16_ddy4(half4 val) {
    return ddy(val);
}

// CHECK-LABEL: define {{.*}} float @_ZN4hlsl8__detail8ddy_implIfEET_S2_
// CHECK: %hlsl.ddy.coarse = call {{.*}} float @llvm.dx.ddy.coarse.f32(float %{{.*}})
// CHECK: ret float %hlsl.ddy.coarse
// CHECK-LABEL-SPIRV: float @_ZN4hlsl8__detail8ddy_implIfEET_S2_
// CHECK-SPIRV: %spv.ddy = call {{.*}} float @llvm.spv.ddy.f32(float %{{.*}})
// CHECK-SPIRV: ret float %spv.ddy
float test_f32_ddy(float val) {
    return ddy(val);
}

// CHECK-LABEL: define {{.*}} <2 x float> @_ZN4hlsl8__detail8ddy_implIDv2_fEET_S3_
// CHECK: %hlsl.ddy.coarse = call {{.*}} <2 x float> @llvm.dx.ddy.coarse.v2f32(<2 x float> %{{.*}})
// CHECK: ret <2 x float> %hlsl.ddy.coarse
// CHECK-LABEL-SPIRV: <2 x float> @_ZN4hlsl8__detail8ddy_implIDv2_fEET_S3_
// CHECK-SPIRV: %spv.ddy = call {{.*}} <2 x float> @llvm.spv.ddy.v2f32(<2 x float> %{{.*}})
// CHECK-SPIRV: ret <2 x float> %spv.ddy
float2 test_f32_ddy2(float2 val) {
    return ddy(val);
}

// CHECK-LABEL: define {{.*}} <3 x float> @_ZN4hlsl8__detail8ddy_implIDv3_fEET_S3_
// CHECK: %hlsl.ddy.coarse = call {{.*}} <3 x float> @llvm.dx.ddy.coarse.v3f32(<3 x float> %{{.*}})
// CHECK: ret <3 x float> %hlsl.ddy.coarse
// CHECK-LABEL-SPIRV: <3 x float> @_ZN4hlsl8__detail8ddy_implIDv3_fEET_S3_
// CHECK-SPIRV: %spv.ddy = call {{.*}} <3 x float> @llvm.spv.ddy.v3f32(<3 x float> %{{.*}})
// CHECK-SPIRV: ret <3 x float> %spv.ddy
float3 test_f32_ddy3(float3 val) {
    return ddy(val);
}

// CHECK-LABEL: define {{.*}} <4 x float> @_ZN4hlsl8__detail8ddy_implIDv4_fEET_S3_
// CHECK: %hlsl.ddy.coarse = call {{.*}} <4 x float> @llvm.dx.ddy.coarse.v4f32(<4 x float> %{{.*}})
// CHECK: ret <4 x float> %hlsl.ddy.coarse
// CHECK-LABEL-SPIRV: <4 x float> @_ZN4hlsl8__detail8ddy_implIDv4_fEET_S3_
// CHECK-SPIRV: %spv.ddy = call {{.*}} <4 x float> @llvm.spv.ddy.v4f32(<4 x float> %{{.*}})
// CHECK-SPIRV: ret <4 x float> %spv.ddy
float4 test_f32_ddy4(float4 val) {
    return ddy(val);
}
