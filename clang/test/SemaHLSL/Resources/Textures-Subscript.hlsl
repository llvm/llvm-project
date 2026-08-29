// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -finclude-default-header -DTEXTURE=Texture2D -DINDEX_TYPE=uint2 \
// RUN:   -DINDEX_ARG="uint2(1, 2)" -DWIDE_INDEX_TYPE=int3 \
// RUN:   -DWIDE_INDEX_ARG="int3(1, 2, 3)" -verify -o - %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -finclude-default-header -DTEXTURE=Texture2DArray -DINDEX_TYPE=uint3 \
// RUN:   -DINDEX_ARG="uint3(1, 2, 0)" -DWIDE_INDEX_TYPE=int4 \
// RUN:   -DWIDE_INDEX_ARG="int4(1, 2, 3, 4)" -DNARROW_INDEX_TYPE=uint2 \
// RUN:   -DNARROW_INDEX_ARG="uint2(1, 2)" -verify -o - %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -finclude-default-header -DTEXTURE=RWTexture2D -DHAS_STORE \
// RUN:   -DINDEX_TYPE=uint2 -DINDEX_ARG="uint2(1, 2)" -DWIDE_INDEX_TYPE=int3 \
// RUN:   -DWIDE_INDEX_ARG="int3(1, 2, 3)" -verify -o - %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -finclude-default-header -DTEXTURE=RWTexture2DArray -DHAS_STORE \
// RUN:   -DINDEX_TYPE=uint3 -DINDEX_ARG="uint3(1, 2, 0)" \
// RUN:   -DWIDE_INDEX_TYPE=int4 -DWIDE_INDEX_ARG="int4(1, 2, 3, 4)" \
// RUN:   -DNARROW_INDEX_TYPE=uint2 -DNARROW_INDEX_ARG="uint2(1, 2)" -verify \
// RUN:   -o - %s
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl \
// RUN:   -finclude-default-header -DTEXTURE=RWTexture2D -DHAS_STORE \
// RUN:   -DINDEX_TYPE=uint2 -DINDEX_ARG="uint2(1, 2)" -DWIDE_INDEX_TYPE=int3 \
// RUN:   -DWIDE_INDEX_ARG="int3(1, 2, 3)" -verify -o - %s
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl \
// RUN:   -finclude-default-header -DTEXTURE=RWTexture2DArray -DHAS_STORE \
// RUN:   -DINDEX_TYPE=uint3 -DINDEX_ARG="uint3(1, 2, 0)" \
// RUN:   -DWIDE_INDEX_TYPE=int4 -DWIDE_INDEX_ARG="int4(1, 2, 3, 4)" \
// RUN:   -DNARROW_INDEX_TYPE=uint2 -DNARROW_INDEX_ARG="uint2(1, 2)" -verify \
// RUN:   -o - %s

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   TEXTURE            resource type name
//   INDEX_TYPE         operator[] index type
//   INDEX_ARG          a literal operator[] index
//   WIDE_INDEX_TYPE    an index with one component too many
//   WIDE_INDEX_ARG     a literal WIDE_INDEX_TYPE index
//   NARROW_INDEX_TYPE  an index with one component too few; only defined for
//                      types whose index has more than one component
//   NARROW_INDEX_ARG   a literal NARROW_INDEX_TYPE index
//   HAS_STORE          defined for writable (UAV) textures, which can
//                      additionally store through operator[]
//
// The diagnostics use `-re` directives so that neither the type name nor the
// index width has to be spelled out per texture type.

TEXTURE<float4> Tex;
TEXTURE<float> Tex2;
TEXTURE<int3> Tex3;

struct S { int a; };

void main() {
  INDEX_TYPE valid_index = INDEX_ARG;

  // Reading through the subscript works for every texture type.
  float4 val1 = Tex[valid_index];

  // The index has to be convertible to the coordinate vector.
  S s = { 1 };
  // expected-error-re@+2 {{no viable overloaded operator[] for type '{{.*}}Texture{{.*}}<float4>'}}
  // expected-note-re@*:* {{candidate function not viable: no known conversion from 'S' to 'vector<unsigned int, {{[0-9]}}>'}}
  float4 val2 = Tex[s];

  // A scalar index is splatted, which changes signedness.
  int i = 1;
  float4 val3 = Tex[i]; // expected-warning-re {{implicit conversion changes signedness: 'int' to 'vector<unsigned int, {{[0-9]}}>' (vector of {{[0-9]}} 'unsigned int' values)}}

  // An index with one component too many is truncated.
  WIDE_INDEX_TYPE big = WIDE_INDEX_ARG;
  // expected-warning-re@+2 {{implicit conversion truncates vector: 'int{{[0-9]}}' (aka 'vector<int, {{[0-9]}}>') to 'vector<unsigned int, {{[0-9]}}>' (vector of {{[0-9]}} 'unsigned int' values)}}
  // expected-warning-re@+1 {{implicit conversion changes signedness: 'int{{[0-9]}}' (aka 'vector<int, {{[0-9]}}>') to 'vector<unsigned int, {{[0-9]}}>' (vector of {{[0-9]}} 'unsigned int' values)}}
  float4 val4 = Tex[big];

#ifdef NARROW_INDEX_TYPE
  // An index with one component too few is rejected.
  NARROW_INDEX_TYPE too_few = NARROW_INDEX_ARG;
  // expected-error-re@+2 {{no viable overloaded operator[] for type '{{.*}}Texture{{.*}}<float4>'}}
  // expected-note-re@*:* {{candidate function not viable: no known conversion from 'uint{{[0-9]}}' (aka 'vector<uint, {{[0-9]}}>') to 'vector<unsigned int, {{[0-9]}}>'}}
  float4 val5 = Tex[too_few];
#endif

  // Storing through the subscript is only possible on a writable texture,
  // whose operator[] returns a non-const reference rather than a const one.
#ifdef HAS_STORE
  // A whole texel...
  Tex[valid_index] = float4(1, 2, 3, 4);

  // ...a single component...
  Tex[valid_index].y = 5.0;

  // ...a swizzle...
  Tex[valid_index].xy = float2(6, 7);

  // ...and a read-modify-write can all be stored through it.
  Tex[valid_index] += float4(1, 1, 1, 1);

  // Scalar and integer element types work the same way.
  Tex2[valid_index] = 8.0;
  Tex3[valid_index] = int3(9, 10, 11);
#else
  // ...whereas every store through a read-only texture is rejected.
  // expected-note@*:* 3 {{function 'operator[]' which returns const-qualified type 'vector<float, 4> const hlsl_device &' declared here}}

  // expected-error@+1 {{cannot assign to return value because function 'operator[]' returns a const value}}
  Tex[valid_index] = float4(1, 2, 3, 4);

  // expected-error@+1 {{cannot assign to return value because function 'operator[]' returns a const value}}
  Tex[valid_index].y = 5.0;

  // expected-error@+1 {{cannot assign to return value because function 'operator[]' returns a const value}}
  Tex[valid_index] += float4(1, 1, 1, 1);
#endif
}
