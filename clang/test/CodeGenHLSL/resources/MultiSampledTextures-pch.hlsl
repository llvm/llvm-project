// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -finclude-default-header -DTEXTURE=Texture2DMS -DLOCATION_TYPE=int2 -emit-pch -o %t %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -finclude-default-header -DTEXTURE=Texture2DMS -DLOCATION_TYPE=int2 -include-pch %t -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s -DTEXTURE=Texture2DMS -DKIND=3

// The Texture2DMS<T, N> sample count is part of the resource handle type, so it
// has to survive serialization: a deserialized handle still lowers to a
// dx.MSTexture with sample count 4.

#ifndef HEADER
#define HEADER

TEXTURE<float4, 4> TMS4;

#else

// CHECK: %"class.hlsl::[[TEXTURE]]" = type { target("dx.MSTexture", <4 x float>, 0, 4, 0, [[KIND]]) }

[numthreads(1, 1, 1)]
void main() {
  float4 V = TMS4.Load((LOCATION_TYPE)0, 0);
  (void)V;
}

#endif
