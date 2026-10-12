// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-pixel -hlsl-entry ps -fsyntax-only -verify %s

// PointSize is only legal as a pixel shader input, never an output. An
// `inout` parameter is currently treated as output-only here, so this is
// rejected (see the sibling vk-builtin-point-size-inout-vertex.hlsl, where
// `inout` on a stage where PointSize is a legal output is accepted).

[shader("pixel")]
float4 ps([[vk::builtin("PointSize")]] inout float size : PSIZE) : SV_Target {
  // expected-error@-1 {{semantic 'PSIZE' is not supported in pixel shader outputs}}
  // expected-error@-2 {{'PointSize' builtin cannot be used as pixel shader outputs}}
  size *= 2.0;
  return float4(size, 0, 0, 1);
}
