// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -emit-llvm -disable-llvm-passes -o - -hlsl-entry main %s -verify -verify-ignore-unexpected=note
// RUN: %clang_cc1 -triple spirv-unknown-vulkan-compute -x hlsl -emit-llvm -disable-llvm-passes -o - -hlsl-entry main %s -verify -verify-ignore-unexpected=note

[numthreads(1,1,1)]
void main(unsigned GI) {
  // expected-error@-1 {{semantic annotations must be present for all parameters of an entry function or patch constant function}}
}
