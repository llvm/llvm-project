// RUN: %clang_cc1 -triple dxil--shadermodel6.6-compute -x hlsl -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// CHECK: %"class.hlsl::RWBuffer" = type { target("dx.TypedBuffer", <4 x float>, 1, 0, 0) }
// CHECK: %"class.hlsl::RWBuffer.0" = type { target("dx.TypedBuffer", float, 1, 0, 0) }
// CHECK: %"class.hlsl::StructuredBuffer" = type { target("dx.RawBuffer", i32, 0, 0) }
// CHECK: %"class.hlsl::RWStructuredBuffer" = type { target("dx.RawBuffer", %struct.S, 1, 0) }
// CHECK: %"class.hlsl::RWBuffer.1" = type { target("dx.TypedBuffer", double, 1, 0, 0) }

// CHECK: @_ZL4U0S0 = internal global %"class.hlsl::RWBuffer" poison, align 4
// CHECK: @_ZL4U5S3 = internal global %"class.hlsl::RWBuffer.0" poison, align 4
// CHECK: @_ZL4T2S2 = internal global %"class.hlsl::StructuredBuffer" poison, align 4
// CHECK: @_ZL4T3S0 = internal global %"class.hlsl::RWStructuredBuffer" poison, align 4
// CHECK: @_ZL5Array = internal global [10 x %"class.hlsl::RWBuffer.1"] poison, align 4

// CHECK: %[[HANDLE:.*]] = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
// CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_v4f32_1_0_0t(
// CHECK-SAME: i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, ptr %{{.*}})
// CHECK: %[[HANDLE_PTR:.*]] = getelementptr inbounds nuw %"class.hlsl::RWBuffer", ptr %this{{[0-9]*}}, i32 0, i32 0
// CHECK: store target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %[[HANDLE]], ptr %[[HANDLE_PTR]], align 4
RWBuffer<float4> U0S0 : register(u0);

// CHECK: %[[HANDLE:.*]] = call target("dx.TypedBuffer", float, 1, 0, 0)
// CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0_0t(
// CHECK-SAME: i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, ptr %{{.*}})
// CHECK: %[[HANDLE_PTR:.*]] = getelementptr inbounds nuw %"class.hlsl::RWBuffer.0", ptr %this{{[0-9]*}}, i32 0, i32 0
// CHECK: store target("dx.TypedBuffer", float, 1, 0, 0) %[[HANDLE]], ptr %[[HANDLE_PTR]], align 4
RWBuffer<float> U5S3 : register(u5, space3);

// CHECK: %[[HANDLE:.*]] = call target("dx.RawBuffer", i32, 0, 0)
// CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_0_0t(
// CHECK-SAME: i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, ptr %{{.*}})
// CHECK: %[[HANDLE_PTR:.*]] = getelementptr inbounds nuw %"class.hlsl::StructuredBuffer", ptr %this{{[0-9]*}}, i32 0, i32 0
// CHECK: store target("dx.RawBuffer", i32, 0, 0) %[[HANDLE]], ptr %[[HANDLE_PTR]], align 4
StructuredBuffer<int> T2S2 : register(t2, space2);

// CHECK: %[[HANDLE:.*]] = call target("dx.RawBuffer", %struct.S, 1, 0)
// CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_s_struct.Ss_1_0t(
// CHECK-SAME: i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, ptr %{{.*}})
// CHECK: %[[HANDLE_PTR:.*]] = getelementptr inbounds nuw %"class.hlsl::RWStructuredBuffer", ptr %this{{[0-9]*}}, i32 0, i32 0
// CHECK: store target("dx.RawBuffer", %struct.S, 1, 0) %[[HANDLE]], ptr %[[HANDLE_PTR]], align 4
struct S {
  float4 f;
  int i;
};
RWStructuredBuffer<S> T3S0 : register(u3);

// Resource array elements are initialized on access; make sure there is no call
// to initialize RWBuffer<double>.
// CHECK-NOT: call target("dx.TypedBuffer", double, 1, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f64_1_0_0t(
RWBuffer<double> Array[10] : register(u4, space0);

[numthreads(4,1,1)]
void main() {
  // Reference Array to ensure it is emitted and we can test that it is initialized
  // to poison, but do not index it.
  // Non-array resources are always emitted because they have a constructor initializer.
  (void)Array;
}
