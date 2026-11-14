// Texture1D
// Texture1D
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -std=hlsl202x -emit-llvm -disable-llvm-passes \
// RUN:   -finclude-default-header -DTEXTURE=Texture1D -o - %s | FileCheck %s \
// RUN:   -DTEXTURE=Texture1D --check-prefixes=CHECK,CHECK-TEXEL \
// RUN:   -DHANDLE_TY='target("dx.Texture", <4 x float>, 0, 0, 0, 1)' \
// RUN:   -DSCALAR_HANDLE_TY='target("dx.Texture", float, 0, 0, 0, 1)'
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -std=hlsl202x \
// RUN:   -emit-llvm -disable-llvm-passes -finclude-default-header \
// RUN:   -DTEXTURE=Texture1D -o - %s | FileCheck %s -DTEXTURE=Texture1D \
// RUN:   --check-prefixes=CHECK,CHECK-TEXEL \
// RUN:   -DHANDLE_TY='target("spirv.Image", float, 0, 2, 0, 0, 1, 0)' \
// RUN:   -DSCALAR_HANDLE_TY='target("spirv.Image", float, 0, 2, 0, 0, 1, 0)'

// Texture1DArray
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -std=hlsl202x -emit-llvm -disable-llvm-passes \
// RUN:   -finclude-default-header -DTEXTURE=Texture1DArray -o - %s | \
// RUN:   FileCheck %s -DTEXTURE=Texture1DArray \
// RUN:   --check-prefixes=CHECK,CHECK-TEXEL \
// RUN:   -DHANDLE_TY='target("dx.Texture", <4 x float>, 0, 0, 0, 6)' \
// RUN:   -DSCALAR_HANDLE_TY='target("dx.Texture", float, 0, 0, 0, 6)'
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -std=hlsl202x \
// RUN:   -emit-llvm -disable-llvm-passes -finclude-default-header \
// RUN:   -DTEXTURE=Texture1DArray -o - %s | FileCheck %s \
// RUN:   -DTEXTURE=Texture1DArray --check-prefixes=CHECK,CHECK-TEXEL \
// RUN:   -DHANDLE_TY='target("spirv.Image", float, 0, 2, 1, 0, 1, 0)' \
// RUN:   -DSCALAR_HANDLE_TY='target("spirv.Image", float, 0, 2, 1, 0, 1, 0)'

// Texture2D
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -std=hlsl202x -emit-llvm -disable-llvm-passes \
// RUN:   -finclude-default-header -DTEXTURE=Texture2D -o - %s | FileCheck %s \
// RUN:   -DTEXTURE=Texture2D --check-prefixes=CHECK,CHECK-TEXEL \
// RUN:   -DHANDLE_TY='target("dx.Texture", <4 x float>, 0, 0, 0, 2)' \
// RUN:   -DSCALAR_HANDLE_TY='target("dx.Texture", float, 0, 0, 0, 2)'
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -std=hlsl202x \
// RUN:   -emit-llvm -disable-llvm-passes -finclude-default-header \
// RUN:   -DTEXTURE=Texture2D -o - %s | FileCheck %s -DTEXTURE=Texture2D \
// RUN:   --check-prefixes=CHECK,CHECK-TEXEL \
// RUN:   -DHANDLE_TY='target("spirv.Image", float, 1, 2, 0, 0, 1, 0)' \
// RUN:   -DSCALAR_HANDLE_TY='target("spirv.Image", float, 1, 2, 0, 0, 1, 0)'

// Texture2DArray
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -std=hlsl202x -emit-llvm -disable-llvm-passes \
// RUN:   -finclude-default-header -DTEXTURE=Texture2DArray -o - %s | \
// RUN:   FileCheck %s -DTEXTURE=Texture2DArray \
// RUN:   --check-prefixes=CHECK,CHECK-TEXEL \
// RUN:   -DHANDLE_TY='target("dx.Texture", <4 x float>, 0, 0, 0, 7)' \
// RUN:   -DSCALAR_HANDLE_TY='target("dx.Texture", float, 0, 0, 0, 7)'
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -std=hlsl202x \
// RUN:   -emit-llvm -disable-llvm-passes -finclude-default-header \
// RUN:   -DTEXTURE=Texture2DArray -o - %s | FileCheck %s \
// RUN:   -DTEXTURE=Texture2DArray --check-prefixes=CHECK,CHECK-TEXEL \
// RUN:   -DHANDLE_TY='target("spirv.Image", float, 1, 2, 1, 0, 1, 0)' \
// RUN:   -DSCALAR_HANDLE_TY='target("spirv.Image", float, 1, 2, 1, 0, 1, 0)'

// Texture3D
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -std=hlsl202x -emit-llvm -disable-llvm-passes \
// RUN:   -finclude-default-header -DTEXTURE=Texture3D -o - %s | FileCheck %s \
// RUN:   -DTEXTURE=Texture3D --check-prefixes=CHECK,CHECK-TEXEL \
// RUN:   -DHANDLE_TY='target("dx.Texture", <4 x float>, 0, 0, 0, 4)' \
// RUN:   -DSCALAR_HANDLE_TY='target("dx.Texture", float, 0, 0, 0, 4)'
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -std=hlsl202x \
// RUN:   -emit-llvm -disable-llvm-passes -finclude-default-header \
// RUN:   -DTEXTURE=Texture3D -o - %s | FileCheck %s -DTEXTURE=Texture3D \
// RUN:   --check-prefixes=CHECK,CHECK-TEXEL \
// RUN:   -DHANDLE_TY='target("spirv.Image", float, 2, 2, 0, 0, 1, 0)' \
// RUN:   -DSCALAR_HANDLE_TY='target("spirv.Image", float, 2, 2, 0, 0, 1, 0)'

// TextureCube
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -std=hlsl202x -emit-llvm -disable-llvm-passes \
// RUN:   -finclude-default-header -DTEXTURE=TextureCube -o - %s | FileCheck \
// RUN:   %s -DTEXTURE=TextureCube --check-prefixes=CHECK,CHECK-NOTEXEL \
// RUN:   -DHANDLE_TY='target("dx.Texture", <4 x float>, 0, 0, 0, 5)' \
// RUN:   -DSCALAR_HANDLE_TY='target("dx.Texture", float, 0, 0, 0, 5)'
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -std=hlsl202x \
// RUN:   -emit-llvm -disable-llvm-passes -finclude-default-header \
// RUN:   -DTEXTURE=TextureCube -o - %s | FileCheck %s -DTEXTURE=TextureCube \
// RUN:   --check-prefixes=CHECK,CHECK-NOTEXEL \
// RUN:   -DHANDLE_TY='target("spirv.Image", float, 3, 2, 0, 0, 1, 0)' \
// RUN:   -DSCALAR_HANDLE_TY='target("spirv.Image", float, 3, 2, 0, 0, 1, 0)'

// TextureCubeArray
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -std=hlsl202x -emit-llvm -disable-llvm-passes \
// RUN:   -finclude-default-header -DTEXTURE=TextureCubeArray -o - %s | \
// RUN:   FileCheck %s -DTEXTURE=TextureCubeArray \
// RUN:   --check-prefixes=CHECK,CHECK-NOTEXEL \
// RUN:   -DHANDLE_TY='target("dx.Texture", <4 x float>, 0, 0, 0, 9)' \
// RUN:   -DSCALAR_HANDLE_TY='target("dx.Texture", float, 0, 0, 0, 9)'
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -std=hlsl202x \
// RUN:   -emit-llvm -disable-llvm-passes -finclude-default-header \
// RUN:   -DTEXTURE=TextureCubeArray -o - %s | FileCheck %s \
// RUN:   -DTEXTURE=TextureCubeArray --check-prefixes=CHECK,CHECK-NOTEXEL \
// RUN:   -DHANDLE_TY='target("spirv.Image", float, 3, 2, 1, 0, 1, 0)' \
// RUN:   -DSCALAR_HANDLE_TY='target("spirv.Image", float, 3, 2, 1, 0, 1, 0)'

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   TEXTURE            resource type name
//   HANDLE_TY          the resource handle type of the float4 texture in the
//                      IR
//   SCALAR_HANDLE_TY   the resource handle type of the float texture in the IR
//
// Check prefixes:
//   TEXEL              the type has integer texel addressing (Load,
//                      operator[], mips), and therefore a `mips` field in its
//                      layout
//   NOTEXEL            the type has no integer texel addressing

// CHECK-TEXEL: %"class.hlsl::[[TEXTURE]]" = type { [[HANDLE_TY]], %"struct.hlsl::[[TEXTURE]]<>::mips_type" }
// CHECK-NOTEXEL: %"class.hlsl::[[TEXTURE]]" = type { [[HANDLE_TY]] }
// CHECK-TEXEL: %"class.hlsl::[[TEXTURE]].0" = type { [[SCALAR_HANDLE_TY]], %"struct.hlsl::[[TEXTURE]]<float>::mips_type" }
// CHECK-NOTEXEL: %"class.hlsl::[[TEXTURE]].0" = type { [[SCALAR_HANDLE_TY]] }

// CHECK: @{{.*}}t1 = internal global %"class.hlsl::[[TEXTURE]]" poison, align {{[0-9]+}}
TEXTURE<> t1;

// CHECK: @{{.*}}t2 = internal global %"class.hlsl::[[TEXTURE]].0" poison, align {{[0-9]+}}
TEXTURE<float> t2;

// CHECK: @{{.*}}t3 = internal global %"class.hlsl::[[TEXTURE]]" poison, align {{[0-9]+}}
TEXTURE t3;

void main() {
}
