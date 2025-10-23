// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -fnative-half-type -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK

using hlsl::ddy_coarse;

// CHECK: define linkonce_odr hidden noundef nofpclass(nan inf) half @
// CHECK: %hlsl.ddy.coarse = call reassoc nnan ninf nsz arcp afn half @llvm.dx.ddy.coarse.f16(half %{{.*}})
// CHECK: ret half %hlsl.ddy.coarse
half test_f16_ddy_coarse(half val) {
    return ddy_coarse(val);
}

// CHECK: define linkonce_odr hidden noundef nofpclass(nan inf) <2 x half> @
// CHECK: %hlsl.ddy.coarse = call reassoc nnan ninf nsz arcp afn <2 x half> @llvm.dx.ddy.coarse.v2f16(<2 x half> %{{.*}})
// CHECK: ret <2 x half> %hlsl.ddy.coarse
half2 test_f16_ddy_coarse2(half2 val) {
    return ddy_coarse(val);
}

// CHECK: define linkonce_odr hidden noundef nofpclass(nan inf) <3 x half> @
// CHECK: %hlsl.ddy.coarse = call reassoc nnan ninf nsz arcp afn <3 x half> @llvm.dx.ddy.coarse.v3f16(<3 x half> %{{.*}})
// CHECK: ret <3 x half> %hlsl.ddy.coarse
half3 test_f16_ddy_coarse3(half3 val) {
    return ddy_coarse(val);
}

// CHECK: define linkonce_odr hidden noundef nofpclass(nan inf) <4 x half> @
// CHECK: %hlsl.ddy.coarse = call reassoc nnan ninf nsz arcp afn <4 x half> @llvm.dx.ddy.coarse.v4f16(<4 x half> %{{.*}})
// CHECK: ret <4 x half> %hlsl.ddy.coarse
half4 test_f16_ddy_coarse4(half4 val) {
    return ddy_coarse(val);
}

// CHECK: define linkonce_odr hidden noundef nofpclass(nan inf) float @
// CHECK: %hlsl.ddy.coarse = call reassoc nnan ninf nsz arcp afn float @llvm.dx.ddy.coarse.f32(float %{{.*}})
// CHECK: ret float %hlsl.ddy.coarse
float test_f32_ddy_coarse(float val) {
    return ddy_coarse(val);
}

// CHECK: define linkonce_odr hidden noundef nofpclass(nan inf) <2 x float> @
// CHECK: %hlsl.ddy.coarse = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.dx.ddy.coarse.v2f32(<2 x float> %{{.*}})
// CHECK: ret <2 x float> %hlsl.ddy.coarse
float2 test_f32_ddy_coarse2(float2 val) {
    return ddy_coarse(val);
}

// CHECK: define linkonce_odr hidden noundef nofpclass(nan inf) <3 x float> @
// CHECK: %hlsl.ddy.coarse = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.dx.ddy.coarse.v3f32(<3 x float> %{{.*}})
// CHECK: ret <3 x float> %hlsl.ddy.coarse
float3 test_f32_ddy_coarse3(float3 val) {
    return ddy_coarse(val);
}

// CHECK: define linkonce_odr hidden noundef nofpclass(nan inf) <4 x float> @
// CHECK: %hlsl.ddy.coarse = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.ddy.coarse.v4f32(<4 x float> %{{.*}})
// CHECK: ret <4 x float> %hlsl.ddy.coarse
float4 test_f32_ddy_coarse4(float4 val) {
    return ddy_coarse(val);
}
