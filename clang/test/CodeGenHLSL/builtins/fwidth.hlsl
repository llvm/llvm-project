// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -fnative-half-type -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple spirv-pc-vulkan-compute  %s \
// RUN:  -emit-llvm -disable-llvm-passes -fnative-half-type -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK-SPIRV

// CHECK-LABEL: define {{.*}} half @_ZN4hlsl8__detail11fwidth_implIDhEET_S2_
// CHECK: %hlsl.ddx.coarse = call {{.*}} half @llvm.dx.ddx.coarse.f16(half %{{.*}})
// CHECK: %{{.*}} = call {{.*}} half @llvm.fabs.f16(half %{{.*}})
// CHECK: %hlsl.ddy.coarse = call {{.*}} half @llvm.dx.ddy.coarse.f16(half %{{.*}})
// CHECK: %{{.*}} = call {{.*}} half @llvm.fabs.f16(half %{{.*}})
// CHECK: %{{.*}} = fadd {{.*}} %{{.*}}, %{{.*}}
// CHECK: ret half %{{.*}}
// CHECK-LABEL-SPIRV: half @_Z15test_f16_fwidthDh
// CHECK-SPIRV: %spv.fwidth = call {{.*}} half @llvm.spv.fwidth.f16(half %{{.*}})
// CHECK-SPIRV: ret half %spv.fwidth
half test_f16_fwidth(half val) {
    return fwidth(val);
}

// CHECK-LABEL: define {{.*}} <2 x half> @_ZN4hlsl8__detail11fwidth_implIDv2_DhEET_S3_
// CHECK: %hlsl.ddx.coarse = call {{.*}} <2 x half> @llvm.dx.ddx.coarse.v2f16(<2 x half> %{{.*}})
// CHECK: %{{.*}} = call {{.*}} <2 x half> @llvm.fabs.v2f16(<2 x half> %{{.*}})
// CHECK: %hlsl.ddy.coarse = call {{.*}} <2 x half> @llvm.dx.ddy.coarse.v2f16(<2 x half> %{{.*}})
// CHECK: %{{.*}} = call {{.*}} <2 x half> @llvm.fabs.v2f16(<2 x half> %{{.*}})
// CHECK: %{{.*}} = fadd {{.*}} %{{.*}}, %{{.*}}
// CHECK: ret <2 x half> %{{.*}}
// CHECK-LABEL-SPIRV: <2 x half> @_Z16test_f16_fwidth2Dv2_Dh
// CHECK-SPIRV: %spv.fwidth = call {{.*}} <2 x half> @llvm.spv.fwidth.v2f16(<2 x half> %{{.*}})
// CHECK-SPIRV: ret <2 x half> %spv.fwidth
half2 test_f16_fwidth2(half2 val) {
    return fwidth(val);
}

// CHECK-LABEL: define {{.*}} <3 x half> @_ZN4hlsl8__detail11fwidth_implIDv3_DhEET_S3_
// CHECK: %hlsl.ddx.coarse = call {{.*}} <3 x half> @llvm.dx.ddx.coarse.v3f16(<3 x half> %{{.*}})
// CHECK: %{{.*}} = call {{.*}} <3 x half> @llvm.fabs.v3f16(<3 x half> %{{.*}})
// CHECK: %hlsl.ddy.coarse = call {{.*}} <3 x half> @llvm.dx.ddy.coarse.v3f16(<3 x half> %{{.*}})
// CHECK: %{{.*}} = call {{.*}} <3 x half> @llvm.fabs.v3f16(<3 x half> %{{.*}})
// CHECK: %{{.*}} = fadd {{.*}} %{{.*}}, %{{.*}}
// CHECK: ret <3 x half> %{{.*}}
// CHECK-LABEL-SPIRV: <3 x half> @_Z16test_f16_fwidth3Dv3_Dh
// CHECK-SPIRV: %spv.fwidth = call {{.*}} <3 x half> @llvm.spv.fwidth.v3f16(<3 x half> %{{.*}})
// CHECK-SPIRV: ret <3 x half> %spv.fwidth
half3 test_f16_fwidth3(half3 val) {
    return fwidth(val);
}

// CHECK-LABEL: define {{.*}} <4 x half> @_ZN4hlsl8__detail11fwidth_implIDv4_DhEET_S3_
// CHECK: %hlsl.ddx.coarse = call {{.*}} <4 x half> @llvm.dx.ddx.coarse.v4f16(<4 x half> %{{.*}})
// CHECK: %{{.*}} = call {{.*}} <4 x half> @llvm.fabs.v4f16(<4 x half> %{{.*}})
// CHECK: %hlsl.ddy.coarse = call {{.*}} <4 x half> @llvm.dx.ddy.coarse.v4f16(<4 x half> %{{.*}})
// CHECK: %{{.*}} = call {{.*}} <4 x half> @llvm.fabs.v4f16(<4 x half> %{{.*}})
// CHECK: %{{.*}} = fadd {{.*}} %{{.*}}, %{{.*}}
// CHECK: ret <4 x half> %{{.*}}
// CHECK-LABEL-SPIRV: <4 x half> @_Z16test_f16_fwidth4Dv4_Dh
// CHECK-SPIRV: %spv.fwidth = call {{.*}} <4 x half> @llvm.spv.fwidth.v4f16(<4 x half> %{{.*}})
// CHECK-SPIRV: ret <4 x half> %spv.fwidth
half4 test_f16_fwidth4(half4 val) {
    return fwidth(val);
}

// CHECK-LABEL: define {{.*}} float @_ZN4hlsl8__detail11fwidth_implIfEET_S2_
// CHECK: %hlsl.ddx.coarse = call {{.*}} float @llvm.dx.ddx.coarse.f32(float %{{.*}})
// CHECK: %{{.*}} = call {{.*}} float @llvm.fabs.f32(float %{{.*}})
// CHECK: %hlsl.ddy.coarse = call {{.*}} float @llvm.dx.ddy.coarse.f32(float %{{.*}})
// CHECK: %{{.*}} = call {{.*}} float @llvm.fabs.f32(float %{{.*}})
// CHECK: %{{.*}} = fadd {{.*}} %{{.*}}, %{{.*}}
// CHECK: ret float %{{.*}}
// CHECK-LABEL-SPIRV: float @_Z15test_f32_fwidthf
// CHECK-SPIRV: %spv.fwidth = call {{.*}} float @llvm.spv.fwidth.f32(float %{{.*}})
// CHECK-SPIRV: ret float %spv.fwidth
float test_f32_fwidth(float val) {
    return fwidth(val);
}

// CHECK-LABEL: define {{.*}} <2 x float> @_ZN4hlsl8__detail11fwidth_implIDv2_fEET_S3_
// CHECK: %hlsl.ddx.coarse = call {{.*}} <2 x float> @llvm.dx.ddx.coarse.v2f32(<2 x float> %{{.*}})
// CHECK: %{{.*}} = call {{.*}} <2 x float> @llvm.fabs.v2f32(<2 x float> %{{.*}})
// CHECK: %hlsl.ddy.coarse = call {{.*}} <2 x float> @llvm.dx.ddy.coarse.v2f32(<2 x float> %{{.*}})
// CHECK: %{{.*}} = call {{.*}} <2 x float> @llvm.fabs.v2f32(<2 x float> %{{.*}})
// CHECK: %{{.*}} = fadd {{.*}} %{{.*}}, %{{.*}}
// CHECK: ret <2 x float> %{{.*}}
// CHECK-LABEL-SPIRV: <2 x float> @_Z16test_f32_fwidth2Dv2_f
// CHECK-SPIRV: %spv.fwidth = call {{.*}} <2 x float> @llvm.spv.fwidth.v2f32(<2 x float> %{{.*}})
// CHECK-SPIRV: ret <2 x float> %spv.fwidth
float2 test_f32_fwidth2(float2 val) {
    return fwidth(val);
}

// CHECK-LABEL: define {{.*}} <3 x float> @_ZN4hlsl8__detail11fwidth_implIDv3_fEET_S3_
// CHECK: %hlsl.ddx.coarse = call {{.*}} <3 x float> @llvm.dx.ddx.coarse.v3f32(<3 x float> %{{.*}})
// CHECK: %{{.*}} = call {{.*}} <3 x float> @llvm.fabs.v3f32(<3 x float> %{{.*}})
// CHECK: %hlsl.ddy.coarse = call {{.*}} <3 x float> @llvm.dx.ddy.coarse.v3f32(<3 x float> %{{.*}})
// CHECK: %{{.*}} = call {{.*}} <3 x float> @llvm.fabs.v3f32(<3 x float> %{{.*}})
// CHECK: %{{.*}} = fadd {{.*}} %{{.*}}, %{{.*}}
// CHECK: ret <3 x float> %{{.*}}
// CHECK-LABEL-SPIRV: <3 x float> @_Z16test_f32_fwidth3Dv3_f
// CHECK-SPIRV: %spv.fwidth = call {{.*}} <3 x float> @llvm.spv.fwidth.v3f32(<3 x float> %{{.*}})
// CHECK-SPIRV: ret <3 x float> %spv.fwidth
float3 test_f32_fwidth3(float3 val) {
    return fwidth(val);
}

// CHECK-LABEL: define {{.*}} <4 x float> @_ZN4hlsl8__detail11fwidth_implIDv4_fEET_S3_
// CHECK: %hlsl.ddx.coarse = call {{.*}} <4 x float> @llvm.dx.ddx.coarse.v4f32(<4 x float> %{{.*}})
// CHECK: %{{.*}} = call {{.*}} <4 x float> @llvm.fabs.v4f32(<4 x float> %{{.*}})
// CHECK: %hlsl.ddy.coarse = call {{.*}} <4 x float> @llvm.dx.ddy.coarse.v4f32(<4 x float> %{{.*}})
// CHECK: %{{.*}} = call {{.*}} <4 x float> @llvm.fabs.v4f32(<4 x float> %{{.*}})
// CHECK: %{{.*}} = fadd {{.*}} %{{.*}}, %{{.*}}
// CHECK: ret <4 x float> %{{.*}}
// CHECK-LABEL-SPIRV: <4 x float> @_Z16test_f32_fwidth4Dv4_f
// CHECK-SPIRV: %spv.fwidth = call {{.*}} <4 x float> @llvm.spv.fwidth.v4f32(<4 x float> %{{.*}})
// CHECK-SPIRV: ret <4 x float> %spv.fwidth
float4 test_f32_fwidth4(float4 val) {
    return fwidth(val);
}
