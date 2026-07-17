// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -finclude-default-header -DTEXTURE=Texture2D -DINDEX=uint2 -DINDEX_INIT="uint2(1, 2)" -DBIG=int3 -DBIG_INIT="int3(1, 2, 3)" -DARRAYED=0 -verify -o - %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -finclude-default-header -DTEXTURE=Texture2DArray -DINDEX=uint3 -DINDEX_INIT="uint3(1, 2, 0)" -DBIG=int4 -DBIG_INIT="int4(1, 2, 3, 4)" -DARRAYED=1 -verify -o - %s

// The diagnostics use `-re` directives: the type name is truncated to 'Texture'
// with `{{.*}}` bridging the varying dimension (keeping the '<float4>' element
// type), and the index dimension is matched with `{{[0-9]}}`, so the same
// assertions apply to every texture type.

TEXTURE<float4> Tex;

struct S { int a; };

void main() {
  INDEX valid_index = INDEX_INIT;
  float4 val1 = Tex[valid_index]; // OK

  S s = { 1 };
  // expected-error-re@+2 {{no viable overloaded operator[] for type 'Texture{{.*}}<float4>'}}
  // expected-note-re@*:* {{candidate function not viable: no known conversion from 'S' to 'vector<unsigned int, {{[0-9]}}>'}}
  float4 val2 = Tex[s];

  int i = 1;
  float4 val3 = Tex[i]; // expected-warning-re {{implicit conversion changes signedness: 'int' to 'vector<unsigned int, {{[0-9]}}>' (vector of {{[0-9]}} 'unsigned int' values)}}

  BIG big = BIG_INIT;
  // expected-warning-re@+2 {{implicit conversion truncates vector: 'int{{[0-9]}}' (aka 'vector<int, {{[0-9]}}>') to 'vector<unsigned int, {{[0-9]}}>' (vector of {{[0-9]}} 'unsigned int' values)}}
  // expected-warning-re@+1 {{implicit conversion changes signedness: 'int{{[0-9]}}' (aka 'vector<int, {{[0-9]}}>') to 'vector<unsigned int, {{[0-9]}}>' (vector of {{[0-9]}} 'unsigned int' values)}}
  float4 val4 = Tex[big];

#if ARRAYED
  // Array textures require a 3-component index, so a 2-component index is
  // rejected. This case does not apply to non-arrayed textures, whose index is
  // already 2 components.
  uint2 too_few = uint2(1, 2);
  // expected-error-re@+2 {{no viable overloaded operator[] for type 'Texture{{.*}}<float4>'}}
  // expected-note@*:* {{candidate function not viable: no known conversion from 'uint2' (aka 'vector<uint, 2>') to 'vector<unsigned int, 3>'}}
  float4 val5 = Tex[too_few];
#endif
}
