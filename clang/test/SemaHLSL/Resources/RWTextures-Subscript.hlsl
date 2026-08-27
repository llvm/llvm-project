// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -finclude-default-header -DTEXTURE=RWTexture2D -DINDEX=uint2 -DINDEX_INIT="uint2(1, 2)" -DARRAYED=0 -verify -o - %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -finclude-default-header -DTEXTURE=RWTexture2DArray -DINDEX=uint3 -DINDEX_INIT="uint3(1, 2, 0)" -DARRAYED=1 -verify -o - %s
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -finclude-default-header -DTEXTURE=RWTexture2D -DINDEX=uint2 -DINDEX_INIT="uint2(1, 2)" -DARRAYED=0 -verify -o - %s
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -finclude-default-header -DTEXTURE=RWTexture2DArray -DINDEX=uint3 -DINDEX_INIT="uint3(1, 2, 0)" -DARRAYED=1 -verify -o - %s

// Writable textures return a non-const reference from operator[], so unlike the
// read-only textures in Textures-Subscript.hlsl they can be assigned through.

TEXTURE<float4> Tex;
TEXTURE<float> Tex2;
TEXTURE<int3> Tex3;

struct S { int a; };

void main() {
  INDEX valid_index = INDEX_INIT;

  // Storing a whole texel is fine.
  Tex[valid_index] = float4(1, 2, 3, 4);

  // So is storing a single component...
  Tex[valid_index].y = 5.0;

  // ...a swizzle...
  Tex[valid_index].xy = float2(6, 7);

  // ...and a read-modify-write.
  Tex[valid_index] += float4(1, 1, 1, 1);

  // Scalar and integer element types work the same way.
  Tex2[valid_index] = 8.0;
  Tex3[valid_index] = int3(9, 10, 11);

  // Reading back through the subscript.
  float4 val = Tex[valid_index];

  // The index has to be convertible to the coordinate vector.
  S s = { 1 };
  // expected-error-re@+2 {{no viable overloaded operator[] for type 'RWTexture{{.*}}<float4>'}}
  // expected-note-re@*:* {{candidate function not viable: no known conversion from 'S' to 'vector<unsigned int, {{[0-9]}}>'}}
  Tex[s] = float4(1, 2, 3, 4);

#if ARRAYED
  // Array textures require a 3-component index.
  uint2 too_few = uint2(1, 2);
  // expected-error-re@+2 {{no viable overloaded operator[] for type 'RWTexture{{.*}}<float4>'}}
  // expected-note@*:* {{candidate function not viable: no known conversion from 'uint2' (aka 'vector<uint, 2>') to 'vector<unsigned int, 3>'}}
  Tex[too_few] = float4(1, 2, 3, 4);
#endif
}
