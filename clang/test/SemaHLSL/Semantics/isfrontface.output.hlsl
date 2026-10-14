// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-pixel -finclude-default-header -x hlsl -verify -o - %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -finclude-default-header -x hlsl -verify -o - %s

float4 main(out bool ff : SV_IsFrontFace) : SV_Target {
// expected-error@-1 {{semantic 'SV_IsFrontFace' is not supported in pixel shader outputs}}
  ff = true; // Written, so the parameter is used.
  return ff ? float4(1, 0, 0, 1) : float4(0, 0, 1, 1);
}
