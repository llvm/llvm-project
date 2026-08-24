// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -DTEXTURE=Texture2DMS -o - %s | FileCheck %s --check-prefix=DXIL -DTEXTURE=Texture2DMS -DDXIL_ARGS="<4 x float>, 0, 0, 0, 3" -DDXIL_ARGS4="<4 x float>, 0, 4, 0, 3"
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -DTEXTURE=Texture2DMS -o - %s | FileCheck %s --check-prefix=SPIRV -DTEXTURE=Texture2DMS -DSPIRV_ARGS="float, 1, 2, 0, 1, 1, 0"

// A multisampled texture lowers to a multisampled resource target type. For
// DXIL this is the dedicated `dx.MSTexture` type carrying the multisampled
// resource kind; for SPIR-V it is a `spirv.Image` with the Multisampled (MS)
// operand set to 1.
//
// The Texture2DMS<T, N> sample count N is the second integer operand of
// dx.MSTexture: it is 0 for the default (T, runtime-determined) and 4 for the
// explicit-count global (TMS4), producing two distinct resource types. SPIR-V
// does not encode the count, so both map to the same spirv.Image.

// DXIL: %"class.hlsl::[[TEXTURE]]" = type { target("dx.MSTexture", [[DXIL_ARGS]]) }
// DXIL: %"class.hlsl::[[TEXTURE]].0" = type { target("dx.MSTexture", [[DXIL_ARGS4]]) }
// SPIRV: %"class.hlsl::[[TEXTURE]]" = type { target("spirv.Image", [[SPIRV_ARGS]]) }
// SPIRV: %"class.hlsl::[[TEXTURE]].0" = type { target("spirv.Image", [[SPIRV_ARGS]]) }

// DXIL: @{{.*}}T = internal global %"class.hlsl::[[TEXTURE]]" poison
// SPIRV: @{{.*}}T = internal global %"class.hlsl::[[TEXTURE]]" poison
TEXTURE<float4> T;

// DXIL: @{{.*}}TMS4 = internal global %"class.hlsl::[[TEXTURE]].0" poison
// SPIRV: @{{.*}}TMS4 = internal global %"class.hlsl::[[TEXTURE]].0" poison
TEXTURE<float4, 4> TMS4;

void main() {}
