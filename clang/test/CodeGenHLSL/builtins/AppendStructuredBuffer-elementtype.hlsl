// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-compute -finclude-default-header -fnative-half-type -emit-llvm -o - %s | FileCheck %s

// CHECK: %"class.hlsl::AppendStructuredBuffer" = type { target("dx.RawBuffer", i16, 1, 0)
// CHECK: %"class.hlsl::AppendStructuredBuffer.0" = type { target("dx.RawBuffer", i16, 1, 0)
// CHECK: %"class.hlsl::AppendStructuredBuffer.1" = type { target("dx.RawBuffer", i32, 1, 0)
// CHECK: %"class.hlsl::AppendStructuredBuffer.2" = type { target("dx.RawBuffer", i32, 1, 0)
// CHECK: %"class.hlsl::AppendStructuredBuffer.3" = type { target("dx.RawBuffer", i64, 1, 0)
// CHECK: %"class.hlsl::AppendStructuredBuffer.4" = type { target("dx.RawBuffer", i64, 1, 0)
// CHECK: %"class.hlsl::AppendStructuredBuffer.5" = type { target("dx.RawBuffer", half, 1, 0) 
// CHECK: %"class.hlsl::AppendStructuredBuffer.6" = type { target("dx.RawBuffer", float, 1, 0)
// CHECK: %"class.hlsl::AppendStructuredBuffer.7" = type { target("dx.RawBuffer", double, 1, 0)
// CHECK: %"class.hlsl::AppendStructuredBuffer.8" = type { target("dx.RawBuffer", <4 x i16>, 1, 0)
// CHECK: %"class.hlsl::AppendStructuredBuffer.9" = type { target("dx.RawBuffer", <3 x i32>, 1, 0)
// CHECK: %"class.hlsl::AppendStructuredBuffer.10" = type { target("dx.RawBuffer", <2 x half>, 1, 0)
// CHECK: %"class.hlsl::AppendStructuredBuffer.11" = type { target("dx.RawBuffer", <3 x float>, 1, 0)

AppendStructuredBuffer<int16_t> BufI16;
AppendStructuredBuffer<uint16_t> BufU16;
AppendStructuredBuffer<int> BufI32;
AppendStructuredBuffer<uint> BufU32;
AppendStructuredBuffer<int64_t> BufI64;
AppendStructuredBuffer<uint64_t> BufU64;
AppendStructuredBuffer<half> BufF16;
AppendStructuredBuffer<float> BufF32;
AppendStructuredBuffer<double> BufF64;
AppendStructuredBuffer< vector<int16_t, 4> > BufI16x4;
AppendStructuredBuffer< vector<uint, 3> > BufU32x3;
AppendStructuredBuffer<half2> BufF16x2;
AppendStructuredBuffer<float3> BufF32x3;
// TODO: AppendStructuredBuffer<snorm half> BufSNormF16;
// TODO: AppendStructuredBuffer<unorm half> BufUNormF16;
// TODO: AppendStructuredBuffer<snorm float> BufSNormF32;
// TODO: AppendStructuredBuffer<unorm float> BufUNormF32;
// TODO: AppendStructuredBuffer<snorm double> BufSNormF64;
// TODO: AppendStructuredBuffer<unorm double> BufUNormF64;

[numthreads(1,1,1)]
void main(int GI : SV_GroupIndex) {
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
