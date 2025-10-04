// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-compute -finclude-default-header -fnative-half-type \
// RUN: -emit-llvm -o - -DRESOURCE=StructuredBuffer %s | FileCheck %s -DRESOURCE=StructuredBuffer -check-prefixes=DXIL-RO

// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-compute -finclude-default-header -fnative-half-type \
// RUN: -emit-llvm -o - -DRESOURCE=StructuredBuffer %s | FileCheck %s -DRESOURCE=StructuredBuffer -check-prefixes=SPV-RO

// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-compute -finclude-default-header -fnative-half-type \
// RUN: -emit-llvm -o - -DRESOURCE=RWStructuredBuffer %s | FileCheck %s -DRESOURCE=RWStructuredBuffer -check-prefixes=DXIL-RW

// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-compute -finclude-default-header -fnative-half-type \
// RUN: -emit-llvm -o - -DRESOURCE=RWStructuredBuffer %s | FileCheck %s -DRESOURCE=RWStructuredBuffer -check-prefixes=SPV-RW

// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-compute -finclude-default-header -fnative-half-type \
// RUN: -emit-llvm -o - -DRESOURCE=AppendStructuredBuffer %s | FileCheck %s -DRESOURCE=AppendStructuredBuffer -check-prefixes=DXIL-RW

// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-compute -finclude-default-header -fnative-half-type \
// RUN: -emit-llvm -o - -DRESOURCE=AppendStructuredBuffer %s | FileCheck %s -DRESOURCE=AppendStructuredBuffer -check-prefixes=SPV-RW

// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-compute -finclude-default-header -fnative-half-type \
// RUN: -emit-llvm -o - -DRESOURCE=ConsumeStructuredBuffer %s | FileCheck %s -DRESOURCE=ConsumeStructuredBuffer -check-prefixes=DXIL-RW

// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-compute -finclude-default-header -fnative-half-type \
// RUN: -emit-llvm -o - -DRESOURCE=ConsumeStructuredBuffer %s | FileCheck %s -DRESOURCE=ConsumeStructuredBuffer -check-prefixes=SPV-RW

// DXIL-RO: %"class.hlsl::[[RESOURCE]]" = type { target("dx.RawBuffer", i16, 0, 0) }
// DXIL-RO: %"class.hlsl::[[RESOURCE]].0" = type { target("dx.RawBuffer", i16, 0, 0) }
// DXIL-RO: %"class.hlsl::[[RESOURCE]].1" = type { target("dx.RawBuffer", i32, 0, 0) }
// DXIL-RO: %"class.hlsl::[[RESOURCE]].2" = type { target("dx.RawBuffer", i32, 0, 0) }
// DXIL-RO: %"class.hlsl::[[RESOURCE]].3" = type { target("dx.RawBuffer", i64, 0, 0) }
// DXIL-RO: %"class.hlsl::[[RESOURCE]].4" = type { target("dx.RawBuffer", i64, 0, 0) }
// DXIL-RO: %"class.hlsl::[[RESOURCE]].5" = type { target("dx.RawBuffer", half, 0, 0) }
// DXIL-RO: %"class.hlsl::[[RESOURCE]].6" = type { target("dx.RawBuffer", float, 0, 0) }
// DXIL-RO: %"class.hlsl::[[RESOURCE]].7" = type { target("dx.RawBuffer", double, 0, 0) }
// DXIL-RO: %"class.hlsl::[[RESOURCE]].8" = type { target("dx.RawBuffer", <4 x i16>, 0, 0) }
// DXIL-RO: %"class.hlsl::[[RESOURCE]].9" = type { target("dx.RawBuffer", <3 x i32>, 0, 0) }
// DXIL-RO: %"class.hlsl::[[RESOURCE]].10" = type { target("dx.RawBuffer", <2 x half>, 0, 0) }
// DXIL-RO: %"class.hlsl::[[RESOURCE]].11" = type { target("dx.RawBuffer", <3 x float>, 0, 0) }
// DXIL-RO: %"class.hlsl::[[RESOURCE]].12" = type { target("dx.RawBuffer", i32, 0, 0) }
// DXIL-RO: %"class.hlsl::[[RESOURCE]].13" = type { target("dx.RawBuffer", <4 x i32>, 0, 0) }

// DXIL-RW: %"class.hlsl::[[RESOURCE]]" = type { target("dx.RawBuffer", i16, 1, 0), target("dx.RawBuffer", i16, 1, 0) }
// DXIL-RW: %"class.hlsl::[[RESOURCE]].0" = type { target("dx.RawBuffer", i16, 1, 0), target("dx.RawBuffer", i16, 1, 0) }
// DXIL-RW: %"class.hlsl::[[RESOURCE]].1" = type { target("dx.RawBuffer", i32, 1, 0), target("dx.RawBuffer", i32, 1, 0) }
// DXIL-RW: %"class.hlsl::[[RESOURCE]].2" = type { target("dx.RawBuffer", i32, 1, 0), target("dx.RawBuffer", i32, 1, 0) }
// DXIL-RW: %"class.hlsl::[[RESOURCE]].3" = type { target("dx.RawBuffer", i64, 1, 0), target("dx.RawBuffer", i64, 1, 0) }
// DXIL-RW: %"class.hlsl::[[RESOURCE]].4" = type { target("dx.RawBuffer", i64, 1, 0), target("dx.RawBuffer", i64, 1, 0) }
// DXIL-RW: %"class.hlsl::[[RESOURCE]].5" = type { target("dx.RawBuffer", half, 1, 0), target("dx.RawBuffer", half, 1, 0) }
// DXIL-RW: %"class.hlsl::[[RESOURCE]].6" = type { target("dx.RawBuffer", float, 1, 0), target("dx.RawBuffer", float, 1, 0) }
// DXIL-RW: %"class.hlsl::[[RESOURCE]].7" = type { target("dx.RawBuffer", double, 1, 0), target("dx.RawBuffer", double, 1, 0) }
// DXIL-RW: %"class.hlsl::[[RESOURCE]].8" = type { target("dx.RawBuffer", <4 x i16>, 1, 0), target("dx.RawBuffer", <4 x i16>, 1, 0) }
// DXIL-RW: %"class.hlsl::[[RESOURCE]].9" = type { target("dx.RawBuffer", <3 x i32>, 1, 0), target("dx.RawBuffer", <3 x i32>, 1, 0) }
// DXIL-RW: %"class.hlsl::[[RESOURCE]].10" = type { target("dx.RawBuffer", <2 x half>, 1, 0), target("dx.RawBuffer", <2 x half>, 1, 0) }
// DXIL-RW: %"class.hlsl::[[RESOURCE]].11" = type { target("dx.RawBuffer", <3 x float>, 1, 0), target("dx.RawBuffer", <3 x float>, 1, 0) }
// DXIL-RW: %"class.hlsl::[[RESOURCE]].12" = type { target("dx.RawBuffer", i32, 1, 0), target("dx.RawBuffer", i32, 1, 0) }
// DXIL-RW: %"class.hlsl::[[RESOURCE]].13" = type { target("dx.RawBuffer", <4 x i32>, 1, 0), target("dx.RawBuffer", <4 x i32>, 1, 0) }

// SPV-RO: %"class.hlsl::[[RESOURCE]]" = type { target("spirv.VulkanBuffer", [0 x i16], 12, 0) }
// SPV-RO: %"class.hlsl::[[RESOURCE]].0" = type { target("spirv.VulkanBuffer", [0 x i16], 12, 0) }
// SPV-RO: %"class.hlsl::[[RESOURCE]].1" = type { target("spirv.VulkanBuffer", [0 x i32], 12, 0) }
// SPV-RO: %"class.hlsl::[[RESOURCE]].2" = type { target("spirv.VulkanBuffer", [0 x i32], 12, 0) }
// SPV-RO: %"class.hlsl::[[RESOURCE]].3" = type { target("spirv.VulkanBuffer", [0 x i64], 12, 0) }
// SPV-RO: %"class.hlsl::[[RESOURCE]].4" = type { target("spirv.VulkanBuffer", [0 x i64], 12, 0) }
// SPV-RO: %"class.hlsl::[[RESOURCE]].5" = type { target("spirv.VulkanBuffer", [0 x half], 12, 0) }
// SPV-RO: %"class.hlsl::[[RESOURCE]].6" = type { target("spirv.VulkanBuffer", [0 x float], 12, 0) }
// SPV-RO: %"class.hlsl::[[RESOURCE]].7" = type { target("spirv.VulkanBuffer", [0 x double], 12, 0) }
// SPV-RO: %"class.hlsl::[[RESOURCE]].8" = type { target("spirv.VulkanBuffer", [0 x <4 x i16>], 12, 0) }
// SPV-RO: %"class.hlsl::[[RESOURCE]].9" = type { target("spirv.VulkanBuffer", [0 x <3 x i32>], 12, 0) }
// SPV-RO: %"class.hlsl::[[RESOURCE]].10" = type { target("spirv.VulkanBuffer", [0 x <2 x half>], 12, 0) }
// SPV-RO: %"class.hlsl::[[RESOURCE]].11" = type { target("spirv.VulkanBuffer", [0 x <3 x float>], 12, 0) }
// SPV-RO: %"class.hlsl::[[RESOURCE]].12" = type { target("spirv.VulkanBuffer", [0 x i32], 12, 0) }
// SPV-RO: %"class.hlsl::[[RESOURCE]].13" = type { target("spirv.VulkanBuffer", [0 x <4 x i32>], 12, 0) }

// SPV-RW: %"class.hlsl::[[RESOURCE]]" = type { target("spirv.VulkanBuffer", [0 x i16], 12, 1), target("spirv.VulkanBuffer", i32, 12, 1) }
// SPV-RW: %"class.hlsl::[[RESOURCE]].0" = type { target("spirv.VulkanBuffer", [0 x i16], 12, 1), target("spirv.VulkanBuffer", i32, 12, 1) }
// SPV-RW: %"class.hlsl::[[RESOURCE]].1" = type { target("spirv.VulkanBuffer", [0 x i32], 12, 1), target("spirv.VulkanBuffer", i32, 12, 1) }
// SPV-RW: %"class.hlsl::[[RESOURCE]].2" = type { target("spirv.VulkanBuffer", [0 x i32], 12, 1), target("spirv.VulkanBuffer", i32, 12, 1) }
// SPV-RW: %"class.hlsl::[[RESOURCE]].3" = type { target("spirv.VulkanBuffer", [0 x i64], 12, 1), target("spirv.VulkanBuffer", i32, 12, 1) }
// SPV-RW: %"class.hlsl::[[RESOURCE]].4" = type { target("spirv.VulkanBuffer", [0 x i64], 12, 1), target("spirv.VulkanBuffer", i32, 12, 1) }
// SPV-RW: %"class.hlsl::[[RESOURCE]].5" = type { target("spirv.VulkanBuffer", [0 x half], 12, 1), target("spirv.VulkanBuffer", i32, 12, 1) }
// SPV-RW: %"class.hlsl::[[RESOURCE]].6" = type { target("spirv.VulkanBuffer", [0 x float], 12, 1), target("spirv.VulkanBuffer", i32, 12, 1) }
// SPV-RW: %"class.hlsl::[[RESOURCE]].7" = type { target("spirv.VulkanBuffer", [0 x double], 12, 1), target("spirv.VulkanBuffer", i32, 12, 1) }
// SPV-RW: %"class.hlsl::[[RESOURCE]].8" = type { target("spirv.VulkanBuffer", [0 x <4 x i16>], 12, 1), target("spirv.VulkanBuffer", i32, 12, 1) }
// SPV-RW: %"class.hlsl::[[RESOURCE]].9" = type { target("spirv.VulkanBuffer", [0 x <3 x i32>], 12, 1), target("spirv.VulkanBuffer", i32, 12, 1) }
// SPV-RW: %"class.hlsl::[[RESOURCE]].10" = type { target("spirv.VulkanBuffer", [0 x <2 x half>], 12, 1), target("spirv.VulkanBuffer", i32, 12, 1) }
// SPV-RW: %"class.hlsl::[[RESOURCE]].11" = type { target("spirv.VulkanBuffer", [0 x <3 x float>], 12, 1), target("spirv.VulkanBuffer", i32, 12, 1) }
// SPV-RW: %"class.hlsl::[[RESOURCE]].12" = type { target("spirv.VulkanBuffer", [0 x i32], 12, 1), target("spirv.VulkanBuffer", i32, 12, 1) }
// SPV-RW: %"class.hlsl::[[RESOURCE]].13" = type { target("spirv.VulkanBuffer", [0 x <4 x i32>], 12, 1), target("spirv.VulkanBuffer", i32, 12, 1) }

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
RESOURCE<bool> BufBool;
RESOURCE<bool4> BufBoolVec;
// TODO: RESOURCE<snorm half> BufSNormF16;
// TODO: RESOURCE<unorm half> BufUNormF16;
// TODO: RESOURCE<snorm float> BufSNormF32;
// TODO: RESOURCE<unorm float> BufUNormF32;
// TODO: RESOURCE<snorm double> BufSNormF64;
// TODO: RESOURCE<unorm double> BufUNormF64;

[numthreads(1,1,1)]
void main() {
}
