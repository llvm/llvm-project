// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -finclude-default-header -o - %s -verify -verify-ignore-unexpected
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-library  -x hlsl -finclude-default-header -o - %s -verify -verify-ignore-unexpected

// expected-error@+1 {{attribute 'SV_Position' only applies to a field or parameter of type 'float/float1/float2/float3/float4'}}
void main(vector<float, 5> a : SV_Position) {
}

// expected-error@+1 {{attribute 'SV_Position' only applies to a field or parameter of type 'float/float1/float2/float3/float4'}}
void main(int2 a : SV_Position) {
}
