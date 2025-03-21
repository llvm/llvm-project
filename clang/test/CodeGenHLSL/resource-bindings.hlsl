// RUN: %clang_cc1 -triple dxil--shadermodel6.6-compute -x hlsl -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// CHECK: define internal void @_init_resource__ZL4U0S0()
// CHECK: [[H:%.*]] = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_v4f32_1_0_0t(i32 0, i32 0, i32 1, i32 0, i1 false)
// CHECK: store target("dx.TypedBuffer", <4 x float>, 1, 0, 0) [[H]], ptr @_ZL4U0S0, align 4
RWBuffer<float4> U0S0 : register(u0);

// CHECK: define internal void @_init_resource__ZL4U5S3()
// CHECK: [[H:%.*]] = call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0_0t(i32 3, i32 5, i32 1, i32 0, i1 false)
// CHECK: store target("dx.TypedBuffer", float, 1, 0, 0) [[H]], ptr @_ZL4U5S3, align 4
RWBuffer<float> U5S3 : register(u5, space3);

// CHECK: define internal void @_init_resource__ZL4T2S2()
// CHECK: [[H:%.*]] = call target("dx.RawBuffer", i32, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_0_0t(i32 2, i32 2, i32 1, i32 0, i1 false)
// CHECK: store target("dx.RawBuffer", i32, 0, 0) [[H]], ptr @_ZL4T2S2, align 4
StructuredBuffer<int> T2S2 : register(t2, space2);
struct S {
  float4 f;
  int i;
};

// CHECK: define internal void @_init_resource__ZL4T3S0()
// CHECK: [[H:%.*]] = call target("dx.RawBuffer", %struct.S, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_s_struct.Ss_0_0t(i32 0, i32 3, i32 1, i32 0, i1 false)
// CHECK: store target("dx.RawBuffer", %struct.S, 0, 0) [[H]], ptr @_ZL4T3S0, align 4
StructuredBuffer<S> T3S0 : register(t3);

// CHECK: define void @main()
// CHECK: call void @_init_resource__ZL4U0S0()
// CHECK: call void @_init_resource__ZL4U5S3()
// CHECK: call void @_init_resource__ZL4T2S2()
// CHECK: call void @_init_resource__ZL4T3S0()

[numthreads(4,1,1)]
void main() {}
