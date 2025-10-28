// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-compute -finclude-default-header -fnative-half-type -emit-llvm -o - %s | FileCheck %s -check-prefixes=DXIL
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-compute -finclude-default-header -fnative-half-type -emit-llvm -o - %s | FileCheck %s -check-prefixes=SPV

struct MyStruct {
  float4 a;
  int2 b;
};

// DXIL: %"class.hlsl::ConsumeStructuredBuffer" = type { target("dx.RawBuffer", i16, 1, 0)
// DXIL: %"class.hlsl::ConsumeStructuredBuffer.0" = type { target("dx.RawBuffer", i16, 1, 0)
// DXIL: %"class.hlsl::ConsumeStructuredBuffer.1" = type { target("dx.RawBuffer", i32, 1, 0)
// DXIL: %"class.hlsl::ConsumeStructuredBuffer.2" = type { target("dx.RawBuffer", i32, 1, 0)
// DXIL: %"class.hlsl::ConsumeStructuredBuffer.3" = type { target("dx.RawBuffer", i64, 1, 0)
// DXIL: %"class.hlsl::ConsumeStructuredBuffer.4" = type { target("dx.RawBuffer", i64, 1, 0)
// DXIL: %"class.hlsl::ConsumeStructuredBuffer.5" = type { target("dx.RawBuffer", half, 1, 0)
// DXIL: %"class.hlsl::ConsumeStructuredBuffer.6" = type { target("dx.RawBuffer", float, 1, 0)
// DXIL: %"class.hlsl::ConsumeStructuredBuffer.7" = type { target("dx.RawBuffer", double, 1, 0)
// DXIL: %"class.hlsl::ConsumeStructuredBuffer.8" = type { target("dx.RawBuffer", <4 x i16>, 1, 0)
// DXIL: %"class.hlsl::ConsumeStructuredBuffer.9" = type { target("dx.RawBuffer", <3 x i32>, 1, 0)
// DXIL: %"class.hlsl::ConsumeStructuredBuffer.10" = type { target("dx.RawBuffer", <2 x half>, 1, 0)
// DXIL: %"class.hlsl::ConsumeStructuredBuffer.11" = type { target("dx.RawBuffer", <3 x float>, 1, 0)
// DXIL: %"class.hlsl::ConsumeStructuredBuffer.12" = type { target("dx.RawBuffer", %struct.MyStruct, 1, 0)
// DXIL: %struct.MyStruct = type <{ <4 x float>, <2 x i32> }>
// DXIL: %"class.hlsl::ConsumeStructuredBuffer.13" = type { target("dx.RawBuffer", i32, 1, 0)
// SPV: %"class.hlsl::ConsumeStructuredBuffer.13" = type { target("spirv.VulkanBuffer", [0 x i32], 12, 1)
// DXIL: %"class.hlsl::ConsumeStructuredBuffer.14" = type { target("dx.RawBuffer", <4 x i32>, 1, 0)
// SPV: %"class.hlsl::ConsumeStructuredBuffer.14" = type { target("spirv.VulkanBuffer", [0 x <4 x i32>], 12, 1)

ConsumeStructuredBuffer<int16_t> BufI16;
ConsumeStructuredBuffer<uint16_t> BufU16;
ConsumeStructuredBuffer<int> BufI32;
ConsumeStructuredBuffer<uint> BufU32;
ConsumeStructuredBuffer<int64_t> BufI64;
ConsumeStructuredBuffer<uint64_t> BufU64;
ConsumeStructuredBuffer<half> BufF16;
ConsumeStructuredBuffer<float> BufF32;
ConsumeStructuredBuffer<double> BufF64;
ConsumeStructuredBuffer< vector<int16_t, 4> > BufI16x4;
ConsumeStructuredBuffer< vector<uint, 3> > BufU32x3;
ConsumeStructuredBuffer<half2> BufF16x2;
ConsumeStructuredBuffer<float3> BufF32x3;
// TODO: ConsumeStructuredBuffer<snorm half> BufSNormF16;
// TODO: ConsumeStructuredBuffer<unorm half> BufUNormF16;
// TODO: ConsumeStructuredBuffer<snorm float> BufSNormF32;
// TODO: ConsumeStructuredBuffer<unorm float> BufUNormF32;
// TODO: ConsumeStructuredBuffer<snorm double> BufSNormF64;
// TODO: ConsumeStructuredBuffer<unorm double> BufUNormF64;
ConsumeStructuredBuffer<MyStruct> BufMyStruct;
ConsumeStructuredBuffer<bool> BufBool;
ConsumeStructuredBuffer<bool4> BufBoolVec;

[numthreads(1,1,1)]
void main(int GI : SV_GroupIndex) {
}
