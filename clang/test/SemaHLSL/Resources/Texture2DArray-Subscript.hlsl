// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -finclude-default-header -verify -o - %s

Texture2DArray<float4> Tex;

struct S { int a; };

void main() {
  uint3 valid_index = uint3(1, 2, 0);
  float4 val1 = Tex[valid_index]; // OK

  S s = { 1 };
  // expected-error@+2 {{no viable overloaded operator[] for type 'Texture2DArray<float4>'}}
  // expected-note@*:* {{candidate function not viable: no known conversion from 'S' to 'vector<unsigned int, 3>'}}
  float4 val2 = Tex[s];

  int i = 1;
  float4 val3 = Tex[i]; // expected-warning {{implicit conversion changes signedness: 'int' to 'vector<unsigned int, 3>' (vector of 3 'unsigned int' values)}}

  int4 i4 = int4(1, 2, 3, 4);
  // expected-warning@+2 {{implicit conversion truncates vector: 'int4' (aka 'vector<int, 4>') to 'vector<unsigned int, 3>' (vector of 3 'unsigned int' values)}}
  // expected-warning@+1 {{implicit conversion changes signedness: 'int4' (aka 'vector<int, 4>') to 'vector<unsigned int, 3>' (vector of 3 'unsigned int' values)}}
  float4 val4 = Tex[i4];

  uint2 too_few = uint2(1, 2);
  // expected-error@+2 {{no viable overloaded operator[] for type 'Texture2DArray<float4>'}}
  // expected-note@*:* {{candidate function not viable: no known conversion from 'uint2' (aka 'vector<uint, 2>') to 'vector<unsigned int, 3>'}}
  float4 val5 = Tex[too_few];
}
