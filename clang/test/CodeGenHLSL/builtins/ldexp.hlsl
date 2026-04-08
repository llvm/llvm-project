// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -fnative-int16-type -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// CHECK-LABEL: define linkonce_odr hidden noundef nofpclass(nan inf) half @_ZN4hlsl5ldexpEDhDh
// CHECK: [[EXP2:%.*]] = call reassoc nnan ninf nsz arcp afn half @llvm.exp2.f16(half %{{.*}})
// CHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn half [[EXP2]], %{{.*}}
// CHECK: ret half [[MUL]]
half test_ldexp_half(half X, half Exp) { return ldexp(X, Exp); }

// CHECK-LABEL: define linkonce_odr hidden noundef nofpclass(nan inf) <2 x half> @_ZN4hlsl5ldexpEDv2_DhS0_
// CHECK: [[EXP2:%.*]] = call reassoc nnan ninf nsz arcp afn <2 x half> @llvm.exp2.v2f16(<2 x half> %{{.*}})
// CHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <2 x half> [[EXP2]], %{{.*}}
// CHECK: ret <2 x half> [[MUL]]
half2 test_ldexp_half2(half2 X, half2 Exp) { return ldexp(X, Exp); }

// CHECK-LABEL: define linkonce_odr hidden noundef nofpclass(nan inf) <3 x half> @_ZN4hlsl5ldexpEDv3_DhS0_
// CHECK: [[EXP2:%.*]] = call reassoc nnan ninf nsz arcp afn <3 x half> @llvm.exp2.v3f16(<3 x half> %{{.*}})
// CHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <3 x half> [[EXP2]], %{{.*}}
// CHECK: ret <3 x half> [[MUL]]
half3 test_ldexp_half3(half3 X, half3 Exp) { return ldexp(X, Exp); }

// CHECK-LABEL: define linkonce_odr hidden noundef nofpclass(nan inf) <4 x half> @_ZN4hlsl5ldexpEDv4_DhS0_
// CHECK: [[EXP2:%.*]] = call reassoc nnan ninf nsz arcp afn <4 x half> @llvm.exp2.v4f16(<4 x half> %{{.*}})
// CHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <4 x half> [[EXP2]], %{{.*}}
// CHECK: ret <4 x half> [[MUL]]
half4 test_ldexp_half4(half4 X, half4 Exp) { return ldexp(X, Exp); }

// CHECK-LABEL: define linkonce_odr hidden noundef nofpclass(nan inf) float @_ZN4hlsl5ldexpEff
// CHECK: [[EXP2:%.*]] = call reassoc nnan ninf nsz arcp afn float @llvm.exp2.f32(float %{{.*}})
// CHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[EXP2]], %{{.*}}
// CHECK: ret float [[MUL]]
float test_ldexp_float(float X, float Exp) { return ldexp(X, Exp); }

// CHECK-LABEL: define linkonce_odr hidden noundef nofpclass(nan inf) <2 x float> @_ZN4hlsl5ldexpEDv2_fS0_
// CHECK: [[EXP2:%.*]] = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.exp2.v2f32(<2 x float> %{{.*}})
// CHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <2 x float> [[EXP2]], %{{.*}}
// CHECK: ret <2 x float> [[MUL]]
float2 test_ldexp_float2(float2 X, float2 Exp) { return ldexp(X, Exp); }

// CHECK-LABEL: define linkonce_odr hidden noundef nofpclass(nan inf) <3 x float> @_ZN4hlsl5ldexpEDv3_fS0_
// CHECK: [[EXP2:%.*]] = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.exp2.v3f32(<3 x float> %{{.*}})
// CHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <3 x float> [[EXP2]], %{{.*}}
// CHECK: ret <3 x float> [[MUL]]
float3 test_ldexp_float3(float3 X, float3 Exp) { return ldexp(X, Exp); }

// CHECK-LABEL: define linkonce_odr hidden noundef nofpclass(nan inf) <4 x float> @_ZN4hlsl5ldexpEDv4_fS0_
// CHECK: [[EXP2:%.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.exp2.v4f32(<4 x float> %{{.*}})
// CHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <4 x float> [[EXP2]], %{{.*}}
// CHECK: ret <4 x float> [[MUL]]
float4 test_ldexp_float4(float4 X, float4 Exp) { return ldexp(X, Exp); }
