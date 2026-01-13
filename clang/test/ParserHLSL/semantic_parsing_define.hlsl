// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -o - %s -verify
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-compute -x hlsl -o - %s -verify

#define SomeDefine SV_IWantAPony

// expected-error@7 {{unknown HLSL semantic 'SV_IWantAPony'}}
void Pony(int GI : SomeDefine) { }
