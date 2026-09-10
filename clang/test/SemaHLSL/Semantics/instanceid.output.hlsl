// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -x hlsl -verify -o - %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -finclude-default-header -x hlsl -verify -o - %s

// SV_InstanceID is a system value only on vertex input, so an out parameter is
// rejected.
[shader("vertex")]
uint main(out uint id : SV_InstanceID) : A {
// expected-error@-1 {{semantic 'SV_InstanceID' does not support output}}
  id = 0;
  return 0;
}
