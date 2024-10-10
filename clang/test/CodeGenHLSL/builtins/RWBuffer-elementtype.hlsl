// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-compute -finclude-default-header -fnative-half-type -emit-llvm -o - %s | FileCheck %s

// CHECK: %"class.hlsl::RWBuffer" = type { target("dx.TypedBuffer", i16, 1, 0, 1), i16 }
// CHECK: %"class.hlsl::RWBuffer.0" = type { target("dx.TypedBuffer", i16, 1, 0, 0), i16 }
// CHECK: %"class.hlsl::RWBuffer.1" = type { target("dx.TypedBuffer", i32, 1, 0, 1), i32 }
// CHECK: %"class.hlsl::RWBuffer.2" = type { target("dx.TypedBuffer", i32, 1, 0, 0), i32 }
// CHECK: %"class.hlsl::RWBuffer.3" = type { target("dx.TypedBuffer", i64, 1, 0, 1), i64 }
// CHECK: %"class.hlsl::RWBuffer.4" = type { target("dx.TypedBuffer", i64, 1, 0, 0), i64 }
// CHECK: %"class.hlsl::RWBuffer.5" = type { target("dx.TypedBuffer", half, 1, 0, 0), half }
// CHECK: %"class.hlsl::RWBuffer.6" = type { target("dx.TypedBuffer", float, 1, 0, 0), float }
// CHECK: %"class.hlsl::RWBuffer.7" = type { target("dx.TypedBuffer", double, 1, 0, 0), double }
// CHECK: %"class.hlsl::RWBuffer.8" = type { target("dx.TypedBuffer", <4 x i16>, 1, 0, 0), <4 x i16> }
// CHECK: %"class.hlsl::RWBuffer.9" = type { target("dx.TypedBuffer", <3 x i32>, 1, 0, 0), <3 x i32> }
// CHECK: %"class.hlsl::RWBuffer.10" = type { target("dx.TypedBuffer", <2 x half>, 1, 0, 0), <2 x half> }
// CHECK: %"class.hlsl::RWBuffer.11" = type { target("dx.TypedBuffer", <3 x float>, 1, 0, 0), <3 x float> }

RWBuffer<int16_t> BufI16;
RWBuffer<uint16_t> BufU16;
RWBuffer<int> BufI32;
RWBuffer<uint> BufU32;
RWBuffer<int64_t> BufI64;
RWBuffer<uint64_t> BufU64;
RWBuffer<half> BufF16;
RWBuffer<float> BufF32;
RWBuffer<double> BufF64;
RWBuffer< vector<int16_t, 4> > BufI16x4;
RWBuffer< vector<uint, 3> > BufU32x3;
RWBuffer<half2> BufF16x2;
RWBuffer<float3> BufF32x3;
// TODO: RWBuffer<snorm half> BufSNormF16; -> 11
// TODO: RWBuffer<unorm half> BufUNormF16; -> 12
// TODO: RWBuffer<snorm float> BufSNormF32; -> 13
// TODO: RWBuffer<unorm float> BufUNormF32; -> 14
// TODO: RWBuffer<snorm double> BufSNormF64; -> 15
// TODO: RWBuffer<unorm double> BufUNormF64; -> 16

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

// CHECK: !{{[0-9]+}} = !{ptr @"?BufI16@@3V?$RWBuffer@F@hlsl@@A", i32 10, i32 2,
// CHECK: !{{[0-9]+}} = !{ptr @"?BufU16@@3V?$RWBuffer@G@hlsl@@A", i32 10, i32 3,
// CHECK: !{{[0-9]+}} = !{ptr @"?BufI32@@3V?$RWBuffer@H@hlsl@@A", i32 10, i32 4,
// CHECK: !{{[0-9]+}} = !{ptr @"?BufU32@@3V?$RWBuffer@I@hlsl@@A", i32 10, i32 5,
// CHECK: !{{[0-9]+}} = !{ptr @"?BufI64@@3V?$RWBuffer@J@hlsl@@A", i32 10, i32 6,
// CHECK: !{{[0-9]+}} = !{ptr @"?BufU64@@3V?$RWBuffer@K@hlsl@@A", i32 10, i32 7,
// CHECK: !{{[0-9]+}} = !{ptr @"?BufF16@@3V?$RWBuffer@$f16@@hlsl@@A", i32 10, i32 8,
// CHECK: !{{[0-9]+}} = !{ptr @"?BufF32@@3V?$RWBuffer@M@hlsl@@A", i32 10, i32 9,
// CHECK: !{{[0-9]+}} = !{ptr @"?BufF64@@3V?$RWBuffer@N@hlsl@@A", i32 10, i32 10,
// CHECK: !{{[0-9]+}} = !{ptr @"?BufI16x4@@3V?$RWBuffer@T?$__vector@F$03@__clang@@@hlsl@@A", i32 10, i32 2,
// CHECK: !{{[0-9]+}} = !{ptr @"?BufU32x3@@3V?$RWBuffer@T?$__vector@I$02@__clang@@@hlsl@@A", i32 10, i32 5,
// CHECK: !{{[0-9]+}} = !{ptr @"?BufF16x2@@3V?$RWBuffer@T?$__vector@$f16@$01@__clang@@@hlsl@@A", i32 10, i32 8,
// CHECK: !{{[0-9]+}} = !{ptr @"?BufF32x3@@3V?$RWBuffer@T?$__vector@M$02@__clang@@@hlsl@@A", i32 10, i32 9,
