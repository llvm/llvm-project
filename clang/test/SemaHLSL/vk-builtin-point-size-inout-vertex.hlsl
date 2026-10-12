// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-vertex -hlsl-entry vs -fsyntax-only -verify %s

// An `inout` PointSize parameter on a vertex shader is accepted, same as
// plain `out` (see the sibling vk-builtin-point-size-inout-pixel.hlsl for a
// stage where `inout`'s current treatment as output-only changes the
// outcome).

[shader("vertex")]
float4 vs([[vk::builtin("PointSize")]] inout float size : PSIZE) : SV_Position {
  size *= 2.0;
  return float4(0, 0, 0, 1);
}

// expected-no-diagnostics
