// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-compute -finclude-default-header -fnative-half-type -emit-llvm -o - %s | FileCheck %s

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
