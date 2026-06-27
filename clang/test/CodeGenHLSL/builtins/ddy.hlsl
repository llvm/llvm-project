// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -fnative-half-type -o - | \
// RUN:  FileCheck %s -DCALL=dx.ddy.coarse -DVAR=hlsl.ddy.coarse
// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple spirv-pc-vulkan-pixel  %s \
// RUN:  -emit-llvm -disable-llvm-passes -fnative-half-type -o - | \
// RUN:  FileCheck %s -DCALL=spv.ddy -DVAR=spv.ddy

// CHECK-LABEL: define {{.*}} half @_ZN4hlsl8__detail8ddy_implIDhEET_S2_
// CHECK: %[[VAR]] = call {{.*}} half @llvm.[[CALL]].f16(half %{{.*}})
// CHECK: ret half %[[VAR]]
half test_f16_ddy(half val) {
    return ddy(val);
}

// CHECK-LABEL: define {{.*}} <2 x half> @_ZN4hlsl8__detail8ddy_implIDv2_DhEET_S3_
// CHECK: %[[VAR]] = call {{.*}} <2 x half> @llvm.[[CALL]].v2f16(<2 x half> %{{.*}})
// CHECK: ret <2 x half> %[[VAR]]
half2 test_f16_ddy2(half2 val) {
    return ddy(val);
}

// CHECK-LABEL: define {{.*}} <3 x half> @_ZN4hlsl8__detail8ddy_implIDv3_DhEET_S3_
// CHECK: %[[VAR]] = call {{.*}} <3 x half> @llvm.[[CALL]].v3f16(<3 x half> %{{.*}})
// CHECK: ret <3 x half> %[[VAR]]
half3 test_f16_ddy3(half3 val) {
    return ddy(val);
}

// CHECK-LABEL: define {{.*}} <4 x half> @_ZN4hlsl8__detail8ddy_implIDv4_DhEET_S3_
// CHECK: %[[VAR]] = call {{.*}} <4 x half> @llvm.[[CALL]].v4f16(<4 x half> %{{.*}})
// CHECK: ret <4 x half> %[[VAR]]
half4 test_f16_ddy4(half4 val) {
    return ddy(val);
}

// CHECK-LABEL: define {{.*}} float @_ZN4hlsl8__detail8ddy_implIfEET_S2_
// CHECK: %[[VAR]] = call {{.*}} float @llvm.[[CALL]].f32(float %{{.*}})
// CHECK: ret float %[[VAR]]
float test_f32_ddy(float val) {
    return ddy(val);
}

// CHECK-LABEL: define {{.*}} <2 x float> @_ZN4hlsl8__detail8ddy_implIDv2_fEET_S3_
// CHECK: %[[VAR]] = call {{.*}} <2 x float> @llvm.[[CALL]].v2f32(<2 x float> %{{.*}})
// CHECK: ret <2 x float> %[[VAR]]
float2 test_f32_ddy2(float2 val) {
    return ddy(val);
}

// CHECK-LABEL: define {{.*}} <3 x float> @_ZN4hlsl8__detail8ddy_implIDv3_fEET_S3_
// CHECK: %[[VAR]] = call {{.*}} <3 x float> @llvm.[[CALL]].v3f32(<3 x float> %{{.*}})
// CHECK: ret <3 x float> %[[VAR]]
float3 test_f32_ddy3(float3 val) {
    return ddy(val);
}

// CHECK-LABEL: define {{.*}} <4 x float> @_ZN4hlsl8__detail8ddy_implIDv4_fEET_S3_
// CHECK: %[[VAR]] = call {{.*}} <4 x float> @llvm.[[CALL]].v4f32(<4 x float> %{{.*}})
// CHECK: ret <4 x float> %[[VAR]]
float4 test_f32_ddy4(float4 val) {
    return ddy(val);
}
