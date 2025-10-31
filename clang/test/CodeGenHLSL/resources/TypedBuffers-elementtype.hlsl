// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-compute -finclude-default-header -fnative-half-type \
// RUN: -emit-llvm -o - -DRESOURCE=Buffer %s | FileCheck %s -DRESOURCE=Buffer -DRW=0 -check-prefixes=DXIL

// RUN: %clang_cc1 -triple spirv-pc-vulkan-compute -finclude-default-header -fnative-half-type \
// RUN: -emit-llvm -o - -DRESOURCE=Buffer %s | FileCheck %s -DRESOURCE=Buffer -DRW=1 -check-prefixes=SPV-RO

// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-compute -finclude-default-header -fnative-half-type \
// RUN: -emit-llvm -o - -DRESOURCE=RWBuffer %s | FileCheck %s -DRESOURCE=RWBuffer -DRW=1 -check-prefixes=DXIL

// RUN: %clang_cc1 -triple spirv-pc-vulkan-compute -finclude-default-header -fnative-half-type \
// RUN: -emit-llvm -o - -DRESOURCE=RWBuffer %s | FileCheck %s -DRESOURCE=RWBuffer --DRW=2 -check-prefixes=SPV-RW

// DXIL: %"class.hlsl::[[RESOURCE]]" = type { target("dx.TypedBuffer", i16, [[RW]], 0, 1) }
// DXIL: %"class.hlsl::[[RESOURCE]].0" = type { target("dx.TypedBuffer", i16, [[RW]], 0, 0) }
// DXIL: %"class.hlsl::[[RESOURCE]].1" = type { target("dx.TypedBuffer", i32, [[RW]], 0, 1) }
// DXIL: %"class.hlsl::[[RESOURCE]].2" = type { target("dx.TypedBuffer", i32, [[RW]], 0, 0) }
// DXIL: %"class.hlsl::[[RESOURCE]].3" = type { target("dx.TypedBuffer", i64, [[RW]], 0, 1) }
// DXIL: %"class.hlsl::[[RESOURCE]].4" = type { target("dx.TypedBuffer", i64, [[RW]], 0, 0) }
// DXIL: %"class.hlsl::[[RESOURCE]].5" = type { target("dx.TypedBuffer", half, [[RW]], 0, 0) }
// DXIL: %"class.hlsl::[[RESOURCE]].6" = type { target("dx.TypedBuffer", float, [[RW]], 0, 0) }
// DXIL: %"class.hlsl::[[RESOURCE]].7" = type { target("dx.TypedBuffer", double, [[RW]], 0, 0) }
// DXIL: %"class.hlsl::[[RESOURCE]].8" = type { target("dx.TypedBuffer", <4 x i16>, [[RW]], 0, 1) }
// DXIL: %"class.hlsl::[[RESOURCE]].9" = type { target("dx.TypedBuffer", <3 x i32>, [[RW]], 0, 0) }
// DXIL: %"class.hlsl::[[RESOURCE]].10" = type { target("dx.TypedBuffer", <2 x half>, [[RW]], 0, 0) }
// DXIL: %"class.hlsl::[[RESOURCE]].11" = type { target("dx.TypedBuffer", <3 x float>, [[RW]], 0, 0) }
// DXIL: %"class.hlsl::[[RESOURCE]].12" = type { target("dx.TypedBuffer", <4 x i32>, [[RW]], 0, 1) }

// SPV-RO: %"class.hlsl::[[RESOURCE]]" = type { target("spirv.SignedImage", i16, 5, 2, 0, 0, 1, 0) }
// SPV-RO: %"class.hlsl::[[RESOURCE]].0" = type { target("spirv.Image", i16, 5, 2, 0, 0, 1, 0) }
// SPV-RO: %"class.hlsl::[[RESOURCE]].1" = type { target("spirv.SignedImage", i32, 5, 2, 0, 0, 1, 0) }
// SPV-RO: %"class.hlsl::[[RESOURCE]].2" = type { target("spirv.Image", i32, 5, 2, 0, 0, 1, 0) }
// SPV-RO: %"class.hlsl::[[RESOURCE]].3" = type { target("spirv.SignedImage", i64, 5, 2, 0, 0, 1, 0) }
// SPV-RO: %"class.hlsl::[[RESOURCE]].4" = type { target("spirv.Image", i64, 5, 2, 0, 0, 1, 0) }
// SPV-RO: %"class.hlsl::[[RESOURCE]].5" = type { target("spirv.Image", half, 5, 2, 0, 0, 1, 0) }
// SPV-RO: %"class.hlsl::[[RESOURCE]].6" = type { target("spirv.Image", float, 5, 2, 0, 0, 1, 0) }
// SPV-RO: %"class.hlsl::[[RESOURCE]].7" = type { target("spirv.Image", double, 5, 2, 0, 0, 1, 0) }
// SPV-RO: %"class.hlsl::[[RESOURCE]].8" = type { target("spirv.SignedImage", i16, 5, 2, 0, 0, 1, 0) }
// SPV-RO: %"class.hlsl::[[RESOURCE]].9" = type { target("spirv.Image", i32, 5, 2, 0, 0, 1, 0) }
// SPV-RO: %"class.hlsl::[[RESOURCE]].10" = type { target("spirv.Image", half, 5, 2, 0, 0, 1, 0) }
// SPV-RO: %"class.hlsl::[[RESOURCE]].11" = type { target("spirv.Image", float, 5, 2, 0, 0, 1, 0) }
// SPV-RO: %"class.hlsl::[[RESOURCE]].12" = type { target("spirv.SignedImage", i32, 5, 2, 0, 0, 1, 0) }

// SPV-RW: %"class.hlsl::[[RESOURCE]]" = type { target("spirv.SignedImage", i16, 5, 2, 0, 0, 2, 0) }
// SPV-RW: %"class.hlsl::[[RESOURCE]].0" = type { target("spirv.Image", i16, 5, 2, 0, 0, 2, 0) }
// SPV-RW: %"class.hlsl::[[RESOURCE]].1" = type { target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) }
// SPV-RW: %"class.hlsl::[[RESOURCE]].2" = type { target("spirv.Image", i32, 5, 2, 0, 0, 2, 33) }
// SPV-RW: %"class.hlsl::[[RESOURCE]].3" = type { target("spirv.SignedImage", i64, 5, 2, 0, 0, 2, 41) }
// SPV-RW: %"class.hlsl::[[RESOURCE]].4" = type { target("spirv.Image", i64, 5, 2, 0, 0, 2, 40) }
// SPV-RW: %"class.hlsl::[[RESOURCE]].5" = type { target("spirv.Image", half, 5, 2, 0, 0, 2, 0) }
// SPV-RW: %"class.hlsl::[[RESOURCE]].6" = type { target("spirv.Image", float, 5, 2, 0, 0, 2, 3) }
// SPV-RW: %"class.hlsl::[[RESOURCE]].7" = type { target("spirv.Image", double, 5, 2, 0, 0, 2, 0) }
// SPV-RW: %"class.hlsl::[[RESOURCE]].8" = type { target("spirv.SignedImage", i16, 5, 2, 0, 0, 2, 0) }
// SPV-RW: %"class.hlsl::[[RESOURCE]].9" = type { target("spirv.Image", i32, 5, 2, 0, 0, 2, 0) }
// SPV-RW: %"class.hlsl::[[RESOURCE]].10" = type { target("spirv.Image", half, 5, 2, 0, 0, 2, 0) }
// SPV-RW: %"class.hlsl::[[RESOURCE]].11" = type { target("spirv.Image", float, 5, 2, 0, 0, 2, 0) }
// SPV-RW: %"class.hlsl::[[RESOURCE]].12" = type { target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 21) }

RESOURCE<int16_t> BufI16;
RESOURCE<uint16_t> BufU16;
RESOURCE<int> BufI32;
RESOURCE<uint> BufU32;
RESOURCE<int64_t> BufI64;
RESOURCE<uint64_t> BufU64;
RESOURCE<half> BufF16;
RESOURCE<float> BufF32;
RESOURCE<double> BufF64;
RESOURCE< vector<int16_t, 4> > BufI16x4;
RESOURCE< vector<uint, 3> > BufU32x3;
RESOURCE<half2> BufF16x2;
RESOURCE<float3> BufF32x3;
RESOURCE<int4> BufI32x4;
// TODO: RESOURCE<snorm half> BufSNormF16; -> 11
// TODO: RESOURCE<unorm half> BufUNormF16; -> 12
// TODO: RESOURCE<snorm float> BufSNormF32; -> 13
// TODO: RESOURCE<unorm float> BufUNormF32; -> 14
// TODO: RESOURCE<snorm double> BufSNormF64; -> 15
// TODO: RESOURCE<unorm double> BufUNormF64; -> 16

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
