// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s -verify -verify-ignore-unexpected=note,error
// RUN: %clang_cc1 -triple spirv-linux-vulkan-library -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s -verify -verify-ignore-unexpected=note,error

[shader("compute")]
[numthreads(8,8,1)]
void foo(uint Idx : SV_GroupThreadID1) {
  // expected-error@-1 {{semantic SV_GroupThreadID does not allow indexing}}
}
