// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -finclude-default-header -o - %s -verify

struct S {
  float4 f0 : SV_Position;
// expected-error@+2 {{semantic annotations must be present for all parameters of an entry function or patch constant function}}
// expected-note@+1 {{'f1' used here}}
  float4 f1;
};

[shader("pixel")]
// expected-note@+1 {{'s' declared here}}
void main(S s) {
}

[shader("pixel")]
// expected-error@+2 {{semantic annotations must be present for all parameters of an entry function or patch constant function}}
// expected-note@+1 {{'f' declared here}}
void main2(float4 p : SV_POSITION, float4 f)
{ }
