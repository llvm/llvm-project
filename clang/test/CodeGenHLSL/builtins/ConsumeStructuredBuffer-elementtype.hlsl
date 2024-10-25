// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-compute -finclude-default-header -fnative-half-type -emit-llvm -o - %s | FileCheck %s

// CHECK: %"class.hlsl::ConsumeStructuredBuffer" = type { target("dx.RawBuffer", i16, 1, 0)
// CHECK: %"class.hlsl::ConsumeStructuredBuffer.0" = type { target("dx.RawBuffer", i16, 1, 0)
// CHECK: %"class.hlsl::ConsumeStructuredBuffer.1" = type { target("dx.RawBuffer", i32, 1, 0)
// CHECK: %"class.hlsl::ConsumeStructuredBuffer.2" = type { target("dx.RawBuffer", i32, 1, 0)
// CHECK: %"class.hlsl::ConsumeStructuredBuffer.3" = type { target("dx.RawBuffer", i64, 1, 0)
// CHECK: %"class.hlsl::ConsumeStructuredBuffer.4" = type { target("dx.RawBuffer", i64, 1, 0)
// CHECK: %"class.hlsl::ConsumeStructuredBuffer.5" = type { target("dx.RawBuffer", half, 1, 0) 
// CHECK: %"class.hlsl::ConsumeStructuredBuffer.6" = type { target("dx.RawBuffer", float, 1, 0)
// CHECK: %"class.hlsl::ConsumeStructuredBuffer.7" = type { target("dx.RawBuffer", double, 1, 0)
// CHECK: %"class.hlsl::ConsumeStructuredBuffer.8" = type { target("dx.RawBuffer", <4 x i16>, 1, 0)
// CHECK: %"class.hlsl::ConsumeStructuredBuffer.9" = type { target("dx.RawBuffer", <3 x i32>, 1, 0)
// CHECK: %"class.hlsl::ConsumeStructuredBuffer.10" = type { target("dx.RawBuffer", <2 x half>, 1, 0)
// CHECK: %"class.hlsl::ConsumeStructuredBuffer.11" = type { target("dx.RawBuffer", <3 x float>, 1, 0)

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
