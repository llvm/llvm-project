// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-vertex -x hlsl -emit-llvm -disable-llvm-passes -o - -hlsl-entry main %s -verify -verify-ignore-unexpected=note
// RUN: %clang_cc1 -triple spirv-unknown-vulkan-vertex -x hlsl -emit-llvm -disable-llvm-passes -o - -hlsl-entry main %s -verify -verify-ignore-unexpected=note

float main(unsigned GI : A) {
  // expected-error@-1 {{semantic annotations must be present for all parameters of an entry function or patch constant function}}
}
