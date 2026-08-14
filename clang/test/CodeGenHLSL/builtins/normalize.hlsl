// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type -fnative-int16-type \
// RUN:   -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,DXCHECK
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-library %s -fnative-half-type -fnative-int16-type \
// RUN:   -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,SPVCHECK

// CHECK-LABEL: test_normalize_half
// DXCHECK: [[DOT:%.*]] = fmul reassoc nnan ninf nsz arcp afn half %{{.*}}, %{{.*}}
// DXCHECK-NEXT: [[RSQRT:%.*]] = call reassoc nnan ninf nsz arcp afn half @llvm.dx.rsqrt.f16(half [[DOT]])
// DXCHECK-NEXT: [[RET:%.*]] = fmul reassoc nnan ninf nsz arcp afn half %{{.*}}, [[RSQRT]]
// SPVCHECK: [[RET:%.*]] = call reassoc nnan ninf nsz arcp afn noundef nofpclass(nan inf) half @llvm.spv.normalize.f16(half %{{.*}})
// CHECK-NEXT: ret half [[RET]]
half test_normalize_half(half p0)
{
    return normalize(p0);
}

// CHECK-LABEL: test_normalize_half2
// DXCHECK: [[DOT:%.*]] = call reassoc nnan ninf nsz arcp afn half @llvm.dx.fdot.v2f16(<2 x half> %{{.*}}, <2 x half> %{{.*}})
// DXCHECK-NEXT: [[RSQRT:%.*]] = call reassoc nnan ninf nsz arcp afn half @llvm.dx.rsqrt.f16(half [[DOT]])
// DXCHECK-NEXT: [[SPLATINSERT:%.*]] = insertelement <2 x half> poison, half [[RSQRT]], i64 0
// DXCHECK-NEXT: [[SPLAT:%.*]] = shufflevector <2 x half> [[SPLATINSERT]], <2 x half> poison, <2 x i32> zeroinitializer
// DXCHECK-NEXT: [[RET:%.*]] = fmul reassoc nnan ninf nsz arcp afn <2 x half> %{{.*}}, [[SPLAT]]
// SPVCHECK: [[RET:%.*]] = call reassoc nnan ninf nsz arcp afn noundef nofpclass(nan inf) <2 x half> @llvm.spv.normalize.v2f16(<2 x half> %{{.*}})
// CHECK-NEXT: ret <2 x half> [[RET]]
half2 test_normalize_half2(half2 p0)
{
    return normalize(p0);
}

// CHECK-LABEL: test_normalize_half3
// DXCHECK: [[DOT:%.*]] = call reassoc nnan ninf nsz arcp afn half @llvm.dx.fdot.v3f16(<3 x half> %{{.*}}, <3 x half> %{{.*}})
// DXCHECK-NEXT: [[RSQRT:%.*]] = call reassoc nnan ninf nsz arcp afn half @llvm.dx.rsqrt.f16(half [[DOT]])
// DXCHECK-NEXT: [[SPLATINSERT:%.*]] = insertelement <3 x half> poison, half [[RSQRT]], i64 0
// DXCHECK-NEXT: [[SPLAT:%.*]] = shufflevector <3 x half> [[SPLATINSERT]], <3 x half> poison, <3 x i32> zeroinitializer
// DXCHECK-NEXT: [[RET:%.*]] = fmul reassoc nnan ninf nsz arcp afn <3 x half> %{{.*}}, [[SPLAT]]
// SPVCHECK: [[RET:%.*]] = call reassoc nnan ninf nsz arcp afn noundef nofpclass(nan inf) <3 x half> @llvm.spv.normalize.v3f16(<3 x half> %{{.*}})
// CHECK-NEXT: ret <3 x half> [[RET]]
half3 test_normalize_half3(half3 p0)
{
    return normalize(p0);
}

// CHECK-LABEL: test_normalize_half4
// DXCHECK: [[DOT:%.*]] = call reassoc nnan ninf nsz arcp afn half @llvm.dx.fdot.v4f16(<4 x half> %{{.*}}, <4 x half> %{{.*}})
// DXCHECK-NEXT: [[RSQRT:%.*]] = call reassoc nnan ninf nsz arcp afn half @llvm.dx.rsqrt.f16(half [[DOT]])
// DXCHECK-NEXT: [[SPLATINSERT:%.*]] = insertelement <4 x half> poison, half [[RSQRT]], i64 0
// DXCHECK-NEXT: [[SPLAT:%.*]] = shufflevector <4 x half> [[SPLATINSERT]], <4 x half> poison, <4 x i32> zeroinitializer
// DXCHECK-NEXT: [[RET:%.*]] = fmul reassoc nnan ninf nsz arcp afn <4 x half> %{{.*}}, [[SPLAT]]
// SPVCHECK: [[RET:%.*]] = call reassoc nnan ninf nsz arcp afn noundef nofpclass(nan inf) <4 x half> @llvm.spv.normalize.v4f16(<4 x half> %{{.*}})
// CHECK-NEXT: ret <4 x half> [[RET]]
half4 test_normalize_half4(half4 p0)
{
    return normalize(p0);
}

// CHECK-LABEL: test_normalize_float
// DXCHECK: [[DOT:%.*]] = fmul reassoc nnan ninf nsz arcp afn float %{{.*}}, %{{.*}}
// DXCHECK-NEXT: [[RSQRT:%.*]] = call reassoc nnan ninf nsz arcp afn float @llvm.dx.rsqrt.f32(float [[DOT]])
// DXCHECK-NEXT: [[RET:%.*]] = fmul reassoc nnan ninf nsz arcp afn float %{{.*}}, [[RSQRT]]
// SPVCHECK: [[RET:%.*]] = call reassoc nnan ninf nsz arcp afn noundef nofpclass(nan inf) float @llvm.spv.normalize.f32(float %{{.*}})
// CHECK-NEXT: ret float [[RET]]
float test_normalize_float(float p0)
{
    return normalize(p0);
}

// CHECK-LABEL: test_normalize_float2
// DXCHECK: [[DOT:%.*]] = call reassoc nnan ninf nsz arcp afn float @llvm.dx.fdot.v2f32(<2 x float> %{{.*}}, <2 x float> %{{.*}})
// DXCHECK-NEXT: [[RSQRT:%.*]] = call reassoc nnan ninf nsz arcp afn float @llvm.dx.rsqrt.f32(float [[DOT]])
// DXCHECK-NEXT: [[SPLATINSERT:%.*]] = insertelement <2 x float> poison, float [[RSQRT]], i64 0
// DXCHECK-NEXT: [[SPLAT:%.*]] = shufflevector <2 x float> [[SPLATINSERT]], <2 x float> poison, <2 x i32> zeroinitializer
// DXCHECK-NEXT: [[RET:%.*]] = fmul reassoc nnan ninf nsz arcp afn <2 x float> %{{.*}}, [[SPLAT]]
// SPVCHECK: [[RET:%.*]] = call reassoc nnan ninf nsz arcp afn noundef nofpclass(nan inf) <2 x float> @llvm.spv.normalize.v2f32(<2 x float> %{{.*}})
// CHECK-NEXT: ret <2 x float> [[RET]]
float2 test_normalize_float2(float2 p0)
{
    return normalize(p0);
}

// CHECK-LABEL: test_normalize_float3
// DXCHECK: [[DOT:%.*]] = call reassoc nnan ninf nsz arcp afn float @llvm.dx.fdot.v3f32(<3 x float> %{{.*}}, <3 x float> %{{.*}})
// DXCHECK-NEXT: [[RSQRT:%.*]] = call reassoc nnan ninf nsz arcp afn float @llvm.dx.rsqrt.f32(float [[DOT]])
// DXCHECK-NEXT: [[SPLATINSERT:%.*]] = insertelement <3 x float> poison, float [[RSQRT]], i64 0
// DXCHECK-NEXT: [[SPLAT:%.*]] = shufflevector <3 x float> [[SPLATINSERT]], <3 x float> poison, <3 x i32> zeroinitializer
// DXCHECK-NEXT: [[RET:%.*]] = fmul reassoc nnan ninf nsz arcp afn <3 x float> %{{.*}}, [[SPLAT]]
// SPVCHECK: [[RET:%.*]] = call reassoc nnan ninf nsz arcp afn noundef nofpclass(nan inf) <3 x float> @llvm.spv.normalize.v3f32(<3 x float> %{{.*}})
// CHECK-NEXT: ret <3 x float> [[RET]]
float3 test_normalize_float3(float3 p0)
{
    return normalize(p0);
}

// CHECK-LABEL: test_normalize_float4
// DXCHECK: [[DOT:%.*]] = call reassoc nnan ninf nsz arcp afn float @llvm.dx.fdot.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}})
// DXCHECK-NEXT: [[RSQRT:%.*]] = call reassoc nnan ninf nsz arcp afn float @llvm.dx.rsqrt.f32(float [[DOT]])
// DXCHECK-NEXT: [[SPLATINSERT:%.*]] = insertelement <4 x float> poison, float [[RSQRT]], i64 0
// DXCHECK-NEXT: [[SPLAT:%.*]] = shufflevector <4 x float> [[SPLATINSERT]], <4 x float> poison, <4 x i32> zeroinitializer
// DXCHECK-NEXT: [[RET:%.*]] = fmul reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}}, [[SPLAT]]
// SPVCHECK: [[RET:%.*]] = call reassoc nnan ninf nsz arcp afn noundef nofpclass(nan inf) <4 x float> @llvm.spv.normalize.v4f32(<4 x float> %{{.*}})
// CHECK-NEXT: ret <4 x float> [[RET]]
float4 test_normalize_float4(float4 p0)
{
    return normalize(p0);
}
