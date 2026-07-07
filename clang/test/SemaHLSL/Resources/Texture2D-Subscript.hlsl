// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -finclude-default-header -verify -o - %s

Texture2D<float4> Tex;

struct S { int a; };

void main() {
  uint2 valid_index = uint2(1, 2);
  float4 val1 = Tex[valid_index]; // OK

  S s = { 1 };
  // expected-error@+2 {{no viable overloaded operator[] for type 'Texture2D<float4>'}}
  // expected-note@*:* {{candidate function not viable: no known conversion from 'S' to 'vector<unsigned int, 2>'}}
  float4 val2 = Tex[s];

  int i = 1;
  float4 val3 = Tex[i]; // expected-warning {{implicit conversion changes signedness: 'int' to 'vector<unsigned int, 2>' (vector of 2 'unsigned int' values)}}

  int3 i3 = int3(1, 2, 3);
  // expected-warning@+2 {{implicit conversion truncates vector: 'int3' (aka 'vector<int, 3>') to 'vector<unsigned int, 2>' (vector of 2 'unsigned int' values)}}
  // expected-warning@+1 {{implicit conversion changes signedness: 'int3' (aka 'vector<int, 3>') to 'vector<unsigned int, 2>' (vector of 2 'unsigned int' values)}}
  float4 val4 = Tex[i3];
}
