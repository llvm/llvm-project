// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-compute -finclude-default-header -fnative-half-type -emit-llvm -o - %s | FileCheck %s

// NOTE: The number in type name and whether the struct is packed or not will mostly
// likely change once subscript operators are properly implemented (llvm/llvm-project#95956) 
// and theinterim field of the contained type is removed.

// CHECK: %"class.hlsl::RWStructuredBuffer" = type <{ target("dx.RawBuffer", i16, 1, 0)
// CHECK: %"class.hlsl::RWStructuredBuffer.0" = type <{ target("dx.RawBuffer", i16, 1, 0)
// CHECK: %"class.hlsl::RWStructuredBuffer.2" = type { target("dx.RawBuffer", i32, 1, 0)
// CHECK: %"class.hlsl::RWStructuredBuffer.3" = type { target("dx.RawBuffer", i32, 1, 0)
// CHECK: %"class.hlsl::RWStructuredBuffer.4" = type { target("dx.RawBuffer", i64, 1, 0)
// CHECK: %"class.hlsl::RWStructuredBuffer.5" = type { target("dx.RawBuffer", i64, 1, 0)
// CHECK: %"class.hlsl::RWStructuredBuffer.6" = type <{ target("dx.RawBuffer", half, 1, 0) 
// CHECK: %"class.hlsl::RWStructuredBuffer.8" = type { target("dx.RawBuffer", float, 1, 0)
// CHECK: %"class.hlsl::RWStructuredBuffer.9" = type { target("dx.RawBuffer", double, 1, 0)
// CHECK: %"class.hlsl::RWStructuredBuffer.10" = type { target("dx.RawBuffer", <4 x i16>, 1, 0)
// CHECK: %"class.hlsl::RWStructuredBuffer.11" = type { target("dx.RawBuffer", <3 x i32>, 1, 0)
// CHECK: %"class.hlsl::RWStructuredBuffer.12" = type { target("dx.RawBuffer", <2 x half>, 1, 0)
// CHECK: %"class.hlsl::RWStructuredBuffer.13" = type { target("dx.RawBuffer", <3 x float>, 1, 0)

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
}

// CHECK: !{{[0-9]+}} = !{ptr @BufI16, i32 10, i32 2,
// CHECK: !{{[0-9]+}} = !{ptr @BufU16, i32 10, i32 3,
// CHECK: !{{[0-9]+}} = !{ptr @BufI32, i32 10, i32 4,
// CHECK: !{{[0-9]+}} = !{ptr @BufU32, i32 10, i32 5,
// CHECK: !{{[0-9]+}} = !{ptr @BufI64, i32 10, i32 6,
// CHECK: !{{[0-9]+}} = !{ptr @BufU64, i32 10, i32 7,
// CHECK: !{{[0-9]+}} = !{ptr @BufF16, i32 10, i32 8,
// CHECK: !{{[0-9]+}} = !{ptr @BufF32, i32 10, i32 9,
// CHECK: !{{[0-9]+}} = !{ptr @BufF64, i32 10, i32 10,
// CHECK: !{{[0-9]+}} = !{ptr @BufI16x4, i32 10, i32 2,
// CHECK: !{{[0-9]+}} = !{ptr @BufU32x3, i32 10, i32 5,
// CHECK: !{{[0-9]+}} = !{ptr @BufF16x2, i32 10, i32 8,
// CHECK: !{{[0-9]+}} = !{ptr @BufF32x3, i32 10, i32 9,
