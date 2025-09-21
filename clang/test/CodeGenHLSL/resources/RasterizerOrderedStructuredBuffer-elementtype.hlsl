// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-compute -finclude-default-header -fnative-half-type -emit-llvm -o - %s | FileCheck %s -check-prefixes=DXIL

struct MyStruct {
  float4 a;
  int2 b;
};

// DXIL: %"class.hlsl::RasterizerOrderedStructuredBuffer" = type { target("dx.RawBuffer", i16, 1, 1) }
// DXIL: %"class.hlsl::RasterizerOrderedStructuredBuffer.0" = type { target("dx.RawBuffer", i16, 1, 1) }
// DXIL: %"class.hlsl::RasterizerOrderedStructuredBuffer.1" = type { target("dx.RawBuffer", i32, 1, 1) }
// DXIL: %"class.hlsl::RasterizerOrderedStructuredBuffer.2" = type { target("dx.RawBuffer", i32, 1, 1) }
// DXIL: %"class.hlsl::RasterizerOrderedStructuredBuffer.3" = type { target("dx.RawBuffer", i64, 1, 1) }
// DXIL: %"class.hlsl::RasterizerOrderedStructuredBuffer.4" = type { target("dx.RawBuffer", i64, 1, 1) }
// DXIL: %"class.hlsl::RasterizerOrderedStructuredBuffer.5" = type { target("dx.RawBuffer", half, 1, 1)  }
// DXIL: %"class.hlsl::RasterizerOrderedStructuredBuffer.6" = type { target("dx.RawBuffer", float, 1, 1) }
// DXIL: %"class.hlsl::RasterizerOrderedStructuredBuffer.7" = type { target("dx.RawBuffer", double, 1, 1) }
// DXIL: %"class.hlsl::RasterizerOrderedStructuredBuffer.8" = type { target("dx.RawBuffer", <4 x i16>, 1, 1) }
// DXIL: %"class.hlsl::RasterizerOrderedStructuredBuffer.9" = type { target("dx.RawBuffer", <3 x i32>, 1, 1) }
// DXIL: %"class.hlsl::RasterizerOrderedStructuredBuffer.10" = type { target("dx.RawBuffer", <2 x half>, 1, 1) }
// DXIL: %"class.hlsl::RasterizerOrderedStructuredBuffer.11" = type { target("dx.RawBuffer", <3 x float>, 1, 1) }
// DXIL: %struct.MyStruct = type <{ <4 x float>, <2 x i32> }>

RasterizerOrderedStructuredBuffer<int16_t> BufI16;
RasterizerOrderedStructuredBuffer<uint16_t> BufU16;
RasterizerOrderedStructuredBuffer<int> BufI32;
RasterizerOrderedStructuredBuffer<uint> BufU32;
RasterizerOrderedStructuredBuffer<int64_t> BufI64;
RasterizerOrderedStructuredBuffer<uint64_t> BufU64;
RasterizerOrderedStructuredBuffer<half> BufF16;
RasterizerOrderedStructuredBuffer<float> BufF32;
RasterizerOrderedStructuredBuffer<double> BufF64;
RasterizerOrderedStructuredBuffer< vector<int16_t, 4> > BufI16x4;
RasterizerOrderedStructuredBuffer< vector<uint, 3> > BufU32x3;
RasterizerOrderedStructuredBuffer<half2> BufF16x2;
RasterizerOrderedStructuredBuffer<float3> BufF32x3;
// TODO: RasterizerOrderedStructuredBuffer<snorm half> BufSNormF16;
// TODO: RasterizerOrderedStructuredBuffer<unorm half> BufUNormF16;
// TODO: RasterizerOrderedStructuredBuffer<snorm float> BufSNormF32;
// TODO: RasterizerOrderedStructuredBuffer<unorm float> BufUNormF32;
// TODO: RasterizerOrderedStructuredBuffer<snorm double> BufSNormF64;
// TODO: RasterizerOrderedStructuredBuffer<unorm double> BufUNormF64;
RasterizerOrderedStructuredBuffer<MyStruct> BufMyStruct;

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
  BufMyStruct[GI] = {{0,0,0,0},{0,0}};
}
