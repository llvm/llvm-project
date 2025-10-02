// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-compute -finclude-default-header -fnative-half-type -emit-llvm -o - %s | FileCheck %s -check-prefixes=CHECK
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-compute -finclude-default-header -fnative-half-type -emit-llvm -o - %s | FileCheck %s -check-prefixes=SPV

// CHECK: %"class.hlsl::RWStructuredBuffer" = type { target("dx.RawBuffer", i16, 1, 0), target("dx.RawBuffer", i16, 1, 0) }
// SPV: %"class.hlsl::RWStructuredBuffer" = type { target("spirv.VulkanBuffer", [0 x i16], 12, 1), target("spirv.VulkanBuffer", i32, 12, 1) }
// CHECK: %"class.hlsl::RWStructuredBuffer.0" = type { target("dx.RawBuffer", i16, 1, 0), target("dx.RawBuffer", i16, 1, 0) }
// SPV: %"class.hlsl::RWStructuredBuffer.0" = type { target("spirv.VulkanBuffer", [0 x i16], 12, 1), target("spirv.VulkanBuffer", i32, 12, 1) }
// CHECK: %"class.hlsl::RWStructuredBuffer.1" = type { target("dx.RawBuffer", i32, 1, 0), target("dx.RawBuffer", i32, 1, 0) }
// SPV: %"class.hlsl::RWStructuredBuffer.1" = type { target("spirv.VulkanBuffer", [0 x i32], 12, 1), target("spirv.VulkanBuffer", i32, 12, 1) }
// CHECK: %"class.hlsl::RWStructuredBuffer.2" = type { target("dx.RawBuffer", i32, 1, 0), target("dx.RawBuffer", i32, 1, 0) }
// SPV: %"class.hlsl::RWStructuredBuffer.2" = type { target("spirv.VulkanBuffer", [0 x i32], 12, 1), target("spirv.VulkanBuffer", i32, 12, 1) }
// CHECK: %"class.hlsl::RWStructuredBuffer.3" = type { target("dx.RawBuffer", i64, 1, 0), target("dx.RawBuffer", i64, 1, 0) }
// SPV: %"class.hlsl::RWStructuredBuffer.3" = type { target("spirv.VulkanBuffer", [0 x i64], 12, 1), target("spirv.VulkanBuffer", i32, 12, 1) }
// CHECK: %"class.hlsl::RWStructuredBuffer.4" = type { target("dx.RawBuffer", i64, 1, 0), target("dx.RawBuffer", i64, 1, 0) }
// SPV: %"class.hlsl::RWStructuredBuffer.4" = type { target("spirv.VulkanBuffer", [0 x i64], 12, 1), target("spirv.VulkanBuffer", i32, 12, 1) }
// CHECK: %"class.hlsl::RWStructuredBuffer.5" = type { target("dx.RawBuffer", half, 1, 0), target("dx.RawBuffer", half, 1, 0) }
// SPV: %"class.hlsl::RWStructuredBuffer.5" = type { target("spirv.VulkanBuffer", [0 x half], 12, 1), target("spirv.VulkanBuffer", i32, 12, 1) }
// CHECK: %"class.hlsl::RWStructuredBuffer.6" = type { target("dx.RawBuffer", float, 1, 0), target("dx.RawBuffer", float, 1, 0) }
// SPV: %"class.hlsl::RWStructuredBuffer.6" = type { target("spirv.VulkanBuffer", [0 x float], 12, 1), target("spirv.VulkanBuffer", i32, 12, 1) }
// CHECK: %"class.hlsl::RWStructuredBuffer.7" = type { target("dx.RawBuffer", double, 1, 0), target("dx.RawBuffer", double, 1, 0) }
// SPV: %"class.hlsl::RWStructuredBuffer.7" = type { target("spirv.VulkanBuffer", [0 x double], 12, 1), target("spirv.VulkanBuffer", i32, 12, 1) }
// CHECK: %"class.hlsl::RWStructuredBuffer.8" = type { target("dx.RawBuffer", <4 x i16>, 1, 0), target("dx.RawBuffer", <4 x i16>, 1, 0) }
// SPV: %"class.hlsl::RWStructuredBuffer.8" = type { target("spirv.VulkanBuffer", [0 x <4 x i16>], 12, 1), target("spirv.VulkanBuffer", i32, 12, 1) }
// CHECK: %"class.hlsl::RWStructuredBuffer.9" = type { target("dx.RawBuffer", <3 x i32>, 1, 0), target("dx.RawBuffer", <3 x i32>, 1, 0) }
// SPV: %"class.hlsl::RWStructuredBuffer.9" = type { target("spirv.VulkanBuffer", [0 x <3 x i32>], 12, 1), target("spirv.VulkanBuffer", i32, 12, 1) }
// CHECK: %"class.hlsl::RWStructuredBuffer.10" = type { target("dx.RawBuffer", <2 x half>, 1, 0), target("dx.RawBuffer", <2 x half>, 1, 0) }
// SPV: %"class.hlsl::RWStructuredBuffer.10" = type { target("spirv.VulkanBuffer", [0 x <2 x half>], 12, 1), target("spirv.VulkanBuffer", i32, 12, 1) }
// CHECK: %"class.hlsl::RWStructuredBuffer.11" = type { target("dx.RawBuffer", <3 x float>, 1, 0), target("dx.RawBuffer", <3 x float>, 1, 0) }
// SPV: %"class.hlsl::RWStructuredBuffer.11" = type { target("spirv.VulkanBuffer", [0 x <3 x float>], 12, 1), target("spirv.VulkanBuffer", i32, 12, 1) }
// CHECK: %"class.hlsl::RWStructuredBuffer.12" = type { target("dx.RawBuffer", i32, 1, 0), target("dx.RawBuffer", i32, 1, 0) }
// SPV: %"class.hlsl::RWStructuredBuffer.12" = type { target("spirv.VulkanBuffer", [0 x i32], 12, 1), target("spirv.VulkanBuffer", i32, 12, 1) }
// CHECK: %"class.hlsl::RWStructuredBuffer.13" = type { target("dx.RawBuffer", <4 x i32>, 1, 0), target("dx.RawBuffer", <4 x i32>, 1, 0) }
// SPV: %"class.hlsl::RWStructuredBuffer.13" = type { target("spirv.VulkanBuffer", [0 x <4 x i32>], 12, 1), target("spirv.VulkanBuffer", i32, 12, 1) }

RWStructuredBuffer<int16_t> BufI16;
RWStructuredBuffer<uint16_t> BufU16;
RWStructuredBuffer<int> BufI32;
RWStructuredBuffer<uint> BufU32;
RWStructuredBuffer<int64_t> BufI64;
RWStructuredBuffer<uint64_t> BufU64;
RWStructuredBuffer<half> BufF16;
RWStructuredBuffer<float> BufF32;
RWStructuredBuffer<double> BufF64;
RWStructuredBuffer< vector<int16_t, 4> > BufI16x4;
RWStructuredBuffer< vector<uint, 3> > BufU32x3;
RWStructuredBuffer<half2> BufF16x2;
RWStructuredBuffer<float3> BufF32x3;
RWStructuredBuffer<bool> BufBool;
RWStructuredBuffer<bool4> BufBoolVec;
// TODO: RWStructuredBuffer<snorm half> BufSNormF16;
// TODO: RWStructuredBuffer<unorm half> BufUNormF16;
// TODO: RWStructuredBuffer<snorm float> BufSNormF32;
// TODO: RWStructuredBuffer<unorm float> BufUNormF32;
// TODO: RWStructuredBuffer<snorm double> BufSNormF64;
// TODO: RWStructuredBuffer<unorm double> BufUNormF64;

[numthreads(1,1,1)]
void main(int GI : SV_GroupIndex) {
  BufI16[GI] = 0;
  BufU16[GI] = 0;
  BufI32[GI] = 0;
  BufU32[GI] = 0;
  BufI64[GI] = 0;
  BufU64[GI] = 0;
  BufF16[GI] = 0;
  BufF32[GI] = 0;
  BufF64[GI] = 0;
  BufI16x4[GI] = 0;
  BufU32x3[GI] = 0;
  BufF16x2[GI] = 0;
  BufF32x3[GI] = 0;
  BufBool[GI] = false;
  BufBool[GI] = false;
}
