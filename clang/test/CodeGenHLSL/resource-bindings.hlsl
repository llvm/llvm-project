// RUN: %clang_cc1 -triple dxil--shadermodel6.6-compute -x hlsl -finclude-default-header -emit-llvm -o - %s | FileCheck %s

// CHECK: define internal void @_init_resource_bindings() {

// CHECK: %U0S0_h = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_v4f32_1_0_0t(i32 0, i32 0, i32 1, i32 0, i1 false)
RWBuffer<float4> U0S0 : register(u0);

// CHECK: %U5S3_h = call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0_0t(i32 3, i32 5, i32 1, i32 0, i1 false)
RWBuffer<float> U5S3 : register(u5, space3);

// CHECK: %T2S2_h = call target("dx.RawBuffer", i32, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_0_0t(i32 2, i32 2, i32 1, i32 0, i1 false)
StructuredBuffer<int> T2S2 : register(t2, space2);
struct S {
  float4 f;
  int i;
};

// CHECK: %T3S0_h = call target("dx.RawBuffer", %struct.S, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_s_struct.Ss_0_0t(i32 0, i32 3, i32 1, i32 0, i1 false)
StructuredBuffer<S> T3S0 : register(t3);
