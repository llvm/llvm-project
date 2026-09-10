// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -x hlsl -verify -o - %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -finclude-default-header -x hlsl -verify -o - %s

[shader("pixel")]
float4 main(out bool ff : SV_IsFrontFace) : SV_Target {
// expected-error@-1 {{semantic 'SV_IsFrontFace' does not support output}}
  ff = true;
  return float4(0, 0, 0, 1);
}
