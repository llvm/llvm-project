// RUN: %clang_cc1 -triple spirv-pc-vulkan-compute -finclude-default-header -fnative-half-type -emit-llvm -o - %s | FileCheck %s

// Signed integers
// CHECK: %"class.hlsl::RWBuffer" = type { target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) }
RWBuffer<int> rwb_int;
// CHECK: %"class.hlsl::RWBuffer.0" = type { target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 25) }
RWBuffer<int2> rwb_int2;
// CHECK: %"class.hlsl::RWBuffer.1" = type { target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) }
RWBuffer<int3> rwb_int3;
// CHECK: %"class.hlsl::RWBuffer.2" = type { target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 21) }
RWBuffer<int4> rwb_int4;

// Unsigned integers
// CHECK: %"class.hlsl::RWBuffer.3" = type { target("spirv.Image", i32, 5, 2, 0, 0, 2, 33) }
RWBuffer<uint> rwb_uint;
// CHECK: %"class.hlsl::RWBuffer.4" = type { target("spirv.Image", i32, 5, 2, 0, 0, 2, 35) }
RWBuffer<uint2> rwb_uint2;
// CHECK: %"class.hlsl::RWBuffer.5" = type { target("spirv.Image", i32, 5, 2, 0, 0, 2, 0) }
RWBuffer<uint3> rwb_uint3;
// CHECK: %"class.hlsl::RWBuffer.6" = type { target("spirv.Image", i32, 5, 2, 0, 0, 2, 30) }
RWBuffer<uint4> rwb_uint4;

// 64-bit integers
// CHECK: %"class.hlsl::RWBuffer.7" = type { target("spirv.SignedImage", i64, 5, 2, 0, 0, 2, 41) }
RWBuffer<int64_t> rwb_i64;
// CHECK: %"class.hlsl::RWBuffer.8" = type { target("spirv.SignedImage", i64, 5, 2, 0, 0, 2, 0) }
RWBuffer<int64_t2> rwb_i64_2;
// CHECK: %"class.hlsl::RWBuffer.9" = type { target("spirv.Image", i64, 5, 2, 0, 0, 2, 40) }
RWBuffer<uint64_t> rwb_u64;
// CHECK: %"class.hlsl::RWBuffer.10" = type { target("spirv.Image", i64, 5, 2, 0, 0, 2, 0) }
RWBuffer<uint64_t2> rwb_u64_2;

// Floats
// CHECK: %"class.hlsl::RWBuffer.11" = type { target("spirv.Image", float, 5, 2, 0, 0, 2, 3) }
RWBuffer<float> rwb_float;
// CHECK: %"class.hlsl::RWBuffer.12" = type { target("spirv.Image", float, 5, 2, 0, 0, 2, 6) }
RWBuffer<float2> rwb_float2;
// CHECK: %"class.hlsl::RWBuffer.13" = type { target("spirv.Image", float, 5, 2, 0, 0, 2, 0) }
RWBuffer<float3> rwb_float3;
// CHECK: %"class.hlsl::RWBuffer.14" = type { target("spirv.Image", float, 5, 2, 0, 0, 2, 1) }
RWBuffer<float4> rwb_float4;

// Other types that should get Unknown format
// CHECK: %"class.hlsl::RWBuffer.15" = type { target("spirv.Image", half, 5, 2, 0, 0, 2, 0) }
RWBuffer<half> rwb_half;
// CHECK: %"class.hlsl::RWBuffer.16" = type { target("spirv.Image", double, 5, 2, 0, 0, 2, 0) }
RWBuffer<double> rwb_double;

// Non-UAV resource
// CHECK: %"class.hlsl::Buffer" = type { target("spirv.SignedImage", i32, 5, 2, 0, 0, 1, 0) }
Buffer<int> b_int;

[numthreads(1,1,1)]
void main(int GI : SV_GroupIndex) {
    rwb_int[GI] = 0;
    rwb_int2[GI] = 0;
    rwb_int3[GI] = 0;
    rwb_int4[GI] = 0;
    rwb_uint[GI] = 0;
    rwb_uint2[GI] = 0;
    rwb_uint3[GI] = 0;
    rwb_uint4[GI] = 0;
    rwb_i64[GI] = 0;
    rwb_i64_2[GI] = 0;
    rwb_u64[GI] = 0;
    rwb_u64_2[GI] = 0;
    rwb_float[GI] = 0;
    rwb_float2[GI] = 0;
    rwb_float3[GI] = 0;
    rwb_float4[GI] = 0;
    rwb_half[GI] = 0;
    rwb_double[GI] = 0;
    int val = b_int[GI];
}
