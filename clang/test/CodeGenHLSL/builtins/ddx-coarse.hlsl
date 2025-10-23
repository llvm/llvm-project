// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -fnative-half-type -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK

using hlsl::ddx_coarse;

// CHECK: define linkonce_odr hidden noundef nofpclass(nan inf) half @
// CHECK: %hlsl.ddx.coarse = call reassoc nnan ninf nsz arcp afn half @llvm.dx.ddx.coarse.f16(half %{{.*}})
// CHECK: ret half %hlsl.ddx.coarse
half test_f16_ddx_coarse(half val) {
    return ddx_coarse(val);
}

// CHECK: define linkonce_odr hidden noundef nofpclass(nan inf) <2 x half> @
// CHECK: %hlsl.ddx.coarse = call reassoc nnan ninf nsz arcp afn <2 x half> @llvm.dx.ddx.coarse.v2f16(<2 x half> %{{.*}})
// CHECK: ret <2 x half> %hlsl.ddx.coarse
half2 test_f16_ddx_coarse2(half2 val) {
    return ddx_coarse(val);
}

// CHECK: define linkonce_odr hidden noundef nofpclass(nan inf) <3 x half> @
// CHECK: %hlsl.ddx.coarse = call reassoc nnan ninf nsz arcp afn <3 x half> @llvm.dx.ddx.coarse.v3f16(<3 x half> %{{.*}})
// CHECK: ret <3 x half> %hlsl.ddx.coarse
half3 test_f16_ddx_coarse3(half3 val) {
    return ddx_coarse(val);
}

// CHECK: define linkonce_odr hidden noundef nofpclass(nan inf) <4 x half> @
// CHECK: %hlsl.ddx.coarse = call reassoc nnan ninf nsz arcp afn <4 x half> @llvm.dx.ddx.coarse.v4f16(<4 x half> %{{.*}})
// CHECK: ret <4 x half> %hlsl.ddx.coarse
half4 test_f16_ddx_coarse4(half4 val) {
    return ddx_coarse(val);
}

// CHECK: define linkonce_odr hidden noundef nofpclass(nan inf) float @
// CHECK: %hlsl.ddx.coarse = call reassoc nnan ninf nsz arcp afn float @llvm.dx.ddx.coarse.f32(float %{{.*}})
// CHECK: ret float %hlsl.ddx.coarse
float test_f32_ddx_coarse(float val) {
    return ddx_coarse(val);
}

// CHECK: define linkonce_odr hidden noundef nofpclass(nan inf) <2 x float> @
// CHECK: %hlsl.ddx.coarse = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.dx.ddx.coarse.v2f32(<2 x float> %{{.*}})
// CHECK: ret <2 x float> %hlsl.ddx.coarse
float2 test_f32_ddx_coarse2(float2 val) {
    return ddx_coarse(val);
}

// CHECK: define linkonce_odr hidden noundef nofpclass(nan inf) <3 x float> @
// CHECK: %hlsl.ddx.coarse = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.dx.ddx.coarse.v3f32(<3 x float> %{{.*}})
// CHECK: ret <3 x float> %hlsl.ddx.coarse
float3 test_f32_ddx_coarse3(float3 val) {
    return ddx_coarse(val);
}

// CHECK: define linkonce_odr hidden noundef nofpclass(nan inf) <4 x float> @
// CHECK: %hlsl.ddx.coarse = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.ddx.coarse.v4f32(<4 x float> %{{.*}})
// CHECK: ret <4 x float> %hlsl.ddx.coarse
float4 test_f32_ddx_coarse4(float4 val) {
    return ddx_coarse(val);
}
