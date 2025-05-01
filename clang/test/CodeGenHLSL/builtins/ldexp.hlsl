// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -emit-llvm -o - | FileCheck %s

// CHECK-LABEL: test_ldexp_half
// CHECK: %elt.exp2.i = call reassoc nnan ninf nsz arcp afn half @llvm.exp2.f16(half %{{.*}})
// CHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn half %elt.exp2.i, %{{.*}}
// CHECK: ret half %mul.i
half test_ldexp_half(half X, half Exp) { return ldexp(X, Exp); }

// CHECK-LABEL: test_ldexp_half2
// CHECK: %elt.exp2.i = call reassoc nnan ninf nsz arcp afn <2 x half> @llvm.exp2.v2f16(<2 x half> %{{.*}})
// CHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn <2 x half> %elt.exp2.i, %{{.*}}
// CHECK: ret <2 x half> %mul.i
half2 test_ldexp_half2(half2 X, half2 Exp) { return ldexp(X, Exp); }

// CHECK-LABEL: test_ldexp_half3
// CHECK: %elt.exp2.i = call reassoc nnan ninf nsz arcp afn <3 x half> @llvm.exp2.v3f16(<3 x half> %{{.*}})
// CHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn <3 x half> %elt.exp2.i, %{{.*}}
// CHECK: ret <3 x half> %mul.i
half3 test_ldexp_half3(half3 X, half3 Exp) { return ldexp(X, Exp); }

// CHECK-LABEL: test_ldexp_half4
// CHECK: %elt.exp2.i = call reassoc nnan ninf nsz arcp afn <4 x half> @llvm.exp2.v4f16(<4 x half> %{{.*}})
// CHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn <4 x half> %elt.exp2.i, %{{.*}}
// CHECK: ret <4 x half> %mul.i
half4 test_ldexp_half4(half4 X, half4 Exp) { return ldexp(X, Exp); }

// CHECK-LABEL: test_ldexp_float
// CHECK: %elt.exp2.i = call reassoc nnan ninf nsz arcp afn float @llvm.exp2.f32(float %{{.*}})
// CHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn float %elt.exp2.i, %{{.*}}
// CHECK: ret float %mul.i
float test_ldexp_float(float X, float Exp) { return ldexp(X, Exp); }

// CHECK-LABEL: test_ldexp_float2
// CHECK: %elt.exp2.i = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.exp2.v2f32(<2 x float> %{{.*}})
// CHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn <2 x float> %elt.exp2.i, %{{.*}}
// CHECK: ret <2 x float> %mul.i
float2 test_ldexp_float2(float2 X, float2 Exp) { return ldexp(X, Exp); }

// CHECK-LABEL: test_ldexp_float3
// CHECK: %elt.exp2.i = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.exp2.v3f32(<3 x float> %{{.*}})
// CHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn <3 x float> %elt.exp2.i, %{{.*}}
// CHECK: ret <3 x float> %mul.i
float3 test_ldexp_float3(float3 X, float3 Exp) { return ldexp(X, Exp); }

// CHECK-LABEL: test_ldexp_float4
// CHECK: %elt.exp2.i = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.exp2.v4f32(<4 x float> %{{.*}})
// CHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn <4 x float> %elt.exp2.i, %{{.*}}
// CHECK: ret <4 x float> %mul.i
float4 test_ldexp_float4(float4 X, float4 Exp) { return ldexp(X, Exp); }
