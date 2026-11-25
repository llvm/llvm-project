// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -fnative-int16-type -emit-llvm -o - | FileCheck %s

// CHECK-LABEL: test_ldexp_half2x3
// CHECK: [[EXP2:%.*]] = call reassoc nnan ninf nsz arcp afn <6 x half> @llvm.exp2.v6f16(<6 x half> %{{.*}})
// CHECK: fmul reassoc nnan ninf nsz arcp afn <6 x half> [[EXP2]], %{{.*}}
half2x3 test_ldexp_half2x3(half2x3 p0, half2x3 p1) { return ldexp(p0, p1); }

// CHECK-LABEL: test_ldexp_half3x4
// CHECK: [[EXP2:%.*]] = call reassoc nnan ninf nsz arcp afn <12 x half> @llvm.exp2.v12f16(<12 x half> %{{.*}})
// CHECK: fmul reassoc nnan ninf nsz arcp afn <12 x half> [[EXP2]], %{{.*}}
half3x4 test_ldexp_half3x4(half3x4 p0, half3x4 p1) { return ldexp(p0, p1); }

// CHECK-LABEL: test_ldexp_half4x4
// CHECK: [[EXP2:%.*]] = call reassoc nnan ninf nsz arcp afn <16 x half> @llvm.exp2.v16f16(<16 x half> %{{.*}})
// CHECK: fmul reassoc nnan ninf nsz arcp afn <16 x half> [[EXP2]], %{{.*}}
half4x4 test_ldexp_half4x4(half4x4 p0, half4x4 p1) { return ldexp(p0, p1); }

// CHECK-LABEL: test_ldexp_float2x3
// CHECK: [[EXP2:%.*]] = call reassoc nnan ninf nsz arcp afn <6 x float> @llvm.exp2.v6f32(<6 x float> %{{.*}})
// CHECK: fmul reassoc nnan ninf nsz arcp afn <6 x float> [[EXP2]], %{{.*}}
float2x3 test_ldexp_float2x3(float2x3 p0, float2x3 p1) { return ldexp(p0, p1); }

// CHECK-LABEL: test_ldexp_float3x4
// CHECK: [[EXP2:%.*]] = call reassoc nnan ninf nsz arcp afn <12 x float> @llvm.exp2.v12f32(<12 x float> %{{.*}})
// CHECK: fmul reassoc nnan ninf nsz arcp afn <12 x float> [[EXP2]], %{{.*}}
float3x4 test_ldexp_float3x4(float3x4 p0, float3x4 p1) { return ldexp(p0, p1); }

// CHECK-LABEL: test_ldexp_float4x4
// CHECK: [[EXP2:%.*]] = call reassoc nnan ninf nsz arcp afn <16 x float> @llvm.exp2.v16f32(<16 x float> %{{.*}})
// CHECK: fmul reassoc nnan ninf nsz arcp afn <16 x float> [[EXP2]], %{{.*}}
float4x4 test_ldexp_float4x4(float4x4 p0, float4x4 p1) { return ldexp(p0, p1); }
