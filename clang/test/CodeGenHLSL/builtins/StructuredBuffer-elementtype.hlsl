// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-compute -finclude-default-header -fnative-half-type -emit-llvm -o - %s | FileCheck %s

// NOTE: The number in type name and whether the struct is packed or not will mostly
// likely change once subscript operators are properly implemented (llvm/llvm-project#95956) 
// and theinterim field of the contained type is removed.

// CHECK: %"class.hlsl::StructuredBuffer" = type <{ target("dx.RawBuffer", i16, 0, 0)
// CHECK: %"class.hlsl::StructuredBuffer.0" = type <{ target("dx.RawBuffer", i16, 0, 0)
// CHECK: %"class.hlsl::StructuredBuffer.2" = type { target("dx.RawBuffer", i32, 0, 0)
// CHECK: %"class.hlsl::StructuredBuffer.3" = type { target("dx.RawBuffer", i32, 0, 0)
// CHECK: %"class.hlsl::StructuredBuffer.4" = type { target("dx.RawBuffer", i64, 0, 0)
// CHECK: %"class.hlsl::StructuredBuffer.5" = type { target("dx.RawBuffer", i64, 0, 0)
// CHECK: %"class.hlsl::StructuredBuffer.6" = type <{ target("dx.RawBuffer", half, 0, 0) 
// CHECK: %"class.hlsl::StructuredBuffer.8" = type { target("dx.RawBuffer", float, 0, 0)
// CHECK: %"class.hlsl::StructuredBuffer.9" = type { target("dx.RawBuffer", double, 0, 0)
// CHECK: %"class.hlsl::StructuredBuffer.10" = type { target("dx.RawBuffer", <4 x i16>, 0, 0)
// CHECK: %"class.hlsl::StructuredBuffer.11" = type { target("dx.RawBuffer", <3 x i32>, 0, 0)
// CHECK: %"class.hlsl::StructuredBuffer.12" = type { target("dx.RawBuffer", <2 x half>, 0, 0)
// CHECK: %"class.hlsl::StructuredBuffer.13" = type { target("dx.RawBuffer", <3 x float>, 0, 0)

StructuredBuffer<int16_t> BufI16;
StructuredBuffer<uint16_t> BufU16;
StructuredBuffer<int> BufI32;
StructuredBuffer<uint> BufU32;
StructuredBuffer<int64_t> BufI64;
StructuredBuffer<uint64_t> BufU64;
StructuredBuffer<half> BufF16;
StructuredBuffer<float> BufF32;
StructuredBuffer<double> BufF64;
StructuredBuffer< vector<int16_t, 4> > BufI16x4;
StructuredBuffer< vector<uint, 3> > BufU32x3;
StructuredBuffer<half2> BufF16x2;
StructuredBuffer<float3> BufF32x3;
// TODO: StructuredBuffer<snorm half> BufSNormF16;
// TODO: StructuredBuffer<unorm half> BufUNormF16;
// TODO: StructuredBuffer<snorm float> BufSNormF32;
// TODO: StructuredBuffer<unorm float> BufUNormF32;
// TODO: StructuredBuffer<snorm double> BufSNormF64;
// TODO: StructuredBuffer<unorm double> BufUNormF64;

[numthreads(1,1,1)]
void main(int GI : SV_GroupIndex) {
  int16_t v1 = BufI16[GI];
  uint16_t v2 = BufU16[GI];
  int v3 = BufI32[GI];
  uint v4 = BufU32[GI];
  int64_t v5 = BufI64[GI];
  uint64_t v6 = BufU64[GI];
  half v7 = BufF16[GI];
  float v8 = BufF32[GI];
  double v9 = BufF64[GI];
  vector<int16_t,4> v10 = BufI16x4[GI];
  vector<int, 3> v11 = BufU32x3[GI];
  half2 v12 = BufF16x2[GI];
  float3 v13 = BufF32x3[GI];
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
